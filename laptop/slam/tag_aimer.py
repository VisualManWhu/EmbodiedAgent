"""Aim the pan-tilt camera at AprilTags during navigation.

The camera used to look straight ahead, so the robot was localized only when
its heading happened to point at a tag (about 10% of the time in the field),
and recovered by rotating the whole chassis — slow, open-loop, and it throws
away the heading it had.

This module decides where the CAMERA should look instead:

  * pose known  -> pick the most useful tag and pan to it. "Useful" means seen
    obliquely (a face-on tag gives a poor heading), in range, and within the
    servo's well-behaved travel.
  * pose lost   -> sweep the pan through a few fixed angles before the caller
    falls back to turning the chassis.

Pure logic: ``update`` returns the pan angle to command (or None to leave the
camera alone). The caller owns the HTTP call and must ignore tag fixes while
``settling`` is true — the Pi reports the commanded angle at once, but the
servo needs a moment to get there, and a pose solved mid-swing is wrong.
"""
import math
from dataclasses import dataclass, field

from .landmark_selector import min_view_distance


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class AimConfig:
    # Servo travel to use. The pan response was measured within +-70 deg and
    # is ~10% non-linear at the ends, so stay well inside.
    pan_limit_rad: float = math.radians(60.0)
    min_range_m: float = 0.4
    good_range_m: float = 1.4           # full score up to here
    # An 11 cm tag is ~26 px wide at 2.6 m. Beyond that the solved position
    # is poor: in the field, fixes on tags 2.5-3 m away moved the pose by
    # 0.4-0.9 m against the previous tag.
    max_range_m: float = 2.6
    max_incidence_rad: float = math.radians(72.0)
    # camera height above the floor; a tag mounted well above it leaves the
    # top of the view when the robot is close (see min_view_distance)
    cam_height_m: float = 0.15
    # leave the current tag only for a clearly better one
    switch_margin: float = 1.3
    retarget_min_rad: float = math.radians(6.0)
    settle_sec: float = 0.6             # servo travel time after a command
    # aimed at a tag, settled, and still not seeing it -> it is occluded or
    # the pose is off: drop it for a while and try another
    look_timeout_sec: float = 2.0
    fail_cooldown_sec: float = 10.0
    # lost-pose sweep: with a ~56 deg lens these cover about +-98 deg
    sweep_rad: tuple = tuple(math.radians(a) for a in (0, 40, -40, 70, -70))
    sweep_dwell_sec: float = 0.9        # look time per angle, after settling


@dataclass
class TagAimer:
    tag_map: dict
    config: AimConfig = field(default_factory=AimConfig)
    pan_cmd: float = 0.0                # last commanded pan (rad)
    target_id: int | None = None
    sweep_exhausted: bool = False
    _cmd_t: float = -1e9
    _target_since: float = 0.0
    _sweep_i: int | None = None
    _failed: dict = field(default_factory=dict)      # tag id -> retry time
    _seen_t: dict = field(default_factory=dict)      # tag id -> last seen

    # ---- queries -------------------------------------------------------

    def settling(self, now: float) -> bool:
        return now < self._cmd_t + self.config.settle_sec

    def score(self, tag_id, pose):
        """``(score, pan_rad)`` for looking at ``tag_id`` from ``pose``;
        score 0 = not worth / not possible."""
        e = self.tag_map[tag_id]
        x, y, yaw = pose
        dx, dy = float(e['x']) - x, float(e['y']) - y
        d = math.hypot(dx, dy)
        cfg = self.config
        if not (cfg.min_range_m <= d <= cfg.max_range_m):
            return 0.0, 0.0
        if d < min_view_distance(e, cfg.cam_height_m):
            return 0.0, 0.0                  # too close: it is above the view
        pan = _wrap(math.atan2(dy, dx) - yaw)
        if abs(pan) > cfg.pan_limit_rad:
            return 0.0, pan
        # viewing angle: between the tag's face normal and the tag->robot ray
        normal = math.radians(float(e.get('yaw_deg', 0.0)))
        inc = abs(_wrap(math.atan2(-dy, -dx) - normal))
        if inc > cfg.max_incidence_rad:
            return 0.0, pan                  # behind it, or too edge-on
        oblique = math.sin(min(inc, math.radians(60.0))) + 0.05
        if d <= cfg.good_range_m:
            rng = 1.0
        else:
            rng = 1.0 - 0.7 * ((d - cfg.good_range_m)
                               / (cfg.max_range_m - cfg.good_range_m))
        aim = 1.0 - 0.3 * abs(pan) / cfg.pan_limit_rad
        return oblique * rng * aim, pan

    # ---- inputs --------------------------------------------------------

    def note_seen(self, tag_ids, now: float):
        for i in tag_ids:
            self._seen_t[i] = now

    def reset_sweep(self):
        """Call after the chassis has turned: the camera sweep is worth
        repeating from the new heading."""
        self._sweep_i = None
        self.sweep_exhausted = False

    def reset(self):
        """The camera was re-centred by someone else (stop / visual search)."""
        self.pan_cmd = 0.0
        self.target_id = None
        self._cmd_t = -1e9
        self.reset_sweep()

    def _command(self, pan: float, now: float):
        self.pan_cmd = pan
        self._cmd_t = now
        return pan

    def update(self, now: float, pose, robot_still: bool):
        """``pose``: ``(x, y, yaw)`` if the robot currently trusts its pose
        (fresh tag fix or valid dead reckoning), else None.

        Returns the pan angle (rad) to command now, or None for no change.
        """
        if not robot_still or self.settling(now):
            return None
        cfg = self.config
        if pose is None:
            return self._sweep(now)

        self.reset_sweep()
        # give up on a tag we are pointed at but cannot see
        if (self.target_id is not None
                and now - self._target_since > cfg.look_timeout_sec
                and now - self._seen_t.get(self.target_id, -1e9)
                > cfg.look_timeout_sec):
            self._failed[self.target_id] = now + cfg.fail_cooldown_sec
            self.target_id = None

        best_id, best_score, best_pan = None, 0.0, 0.0
        cur_score, cur_pan = 0.0, 0.0
        for tag_id in self.tag_map:
            if self._failed.get(tag_id, 0.0) > now:
                continue
            s, pan = self.score(tag_id, pose)
            if tag_id == self.target_id:
                cur_score, cur_pan = s, pan
            if s > best_score:
                best_id, best_score, best_pan = tag_id, s, pan

        if best_id is None:
            # nothing worth looking at: face forward (also the best bet for
            # stumbling on a tag as the robot drives)
            self.target_id = None
            return self._retarget(0.0, now)
        if (self.target_id is not None and cur_score > 0.0
                and best_score < cur_score * cfg.switch_margin):
            return self._retarget(cur_pan, now)      # stay on the current tag
        if best_id != self.target_id:
            self.target_id = best_id
            self._target_since = now
        return self._retarget(best_pan, now)

    def _retarget(self, pan: float, now: float):
        if abs(pan - self.pan_cmd) < self.config.retarget_min_rad:
            return None
        return self._command(pan, now)

    def _sweep(self, now: float):
        cfg = self.config
        self.target_id = None
        if self.sweep_exhausted:
            return None
        if self._sweep_i is None:
            self._sweep_i = 0
        elif now < self._cmd_t + cfg.settle_sec + cfg.sweep_dwell_sec:
            return None                              # still looking here
        else:
            self._sweep_i += 1
        if self._sweep_i >= len(cfg.sweep_rad):
            self.sweep_exhausted = True
            return None
        # always (re)issue the sweep angle, even if equal to the current pan,
        # so the dwell timer restarts
        return self._command(cfg.sweep_rad[self._sweep_i], now)
