"""Single-goto state machine.

Consumes a robot pose (real or dead-reckoned) and produces motion pulses
the Pi can execute one at a time. Handles avoidance side-steps from Pi
``blocked:<side>`` events and rotates in place to recover lost localization.
"""
import math
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    INIT = 'INIT'
    DRIVING = 'DRIVING'
    WAITING_PULSE = 'WAITING_PULSE'
    SCANNING = 'SCANNING'
    ARRIVED = 'ARRIVED'
    FAILED = 'FAILED'


@dataclass
class NavConfig:
    arrived_radius_m: float = 0.4
    # >this heading error -> a dedicated in-place rotate. Kept loose: the
    # robot only needs to be ROUGHLY aimed, then drives forward while the
    # blended steering (steer_gain) does the fine correction. Tighter than
    # this and open-loop mecanum rotation (no encoder) plus noisy single-tag
    # yaw can never land inside the tolerance -> the robot spins forever.
    heading_tol_rad: float = 0.5
    v_max: float = 0.15
    w_max: float = 0.5                       # commanded yaw rate (rotate/scan)
    # effective yaw rate the robot ACTUALLY turns at when commanded w_max —
    # open-loop, no encoder, so the commanded number is not the real rate.
    # Used to compute rotate-pulse durations. The kit's hand-measured
    # rotate_90_sec (90 deg / 1.4 s ~= 1.12 rad/s) proved too low in the
    # field: four turns checked against tag fixes all overshot by 40-45%
    # (e.g. a 1.11 s pulse meant for 71 deg turned 103 deg), i.e. ~1.5 rad/s.
    # It varies with battery voltage and floor, so treat it as approximate —
    # a tag fix after the turn is what actually settles the heading.
    w_eff: float = 1.5
    max_pulse_sec: float = 2.5
    min_pulse_sec: float = 0.2
    # forward pulses blend in proportional steering so the robot curves onto
    # the goal line instead of drifting off and snap-correcting later.
    steer_gain: float = 1.5                  # wz per rad of heading error
    forward_steer_max: float = 0.25          # cap on the blended wz (rad/s)
    # still time after each drive pulse so the detector gets sharp,
    # motion-blur-free frames to re-localize before the next pulse.
    nav_dwell_sec: float = 2.0
    strafe_speed: float = 0.15
    # Side-step per block. The Nth consecutive block strafes N x this long
    # (capped), i.e. ~15 cm, then ~30 cm: a fixed 9 cm step three times in a
    # row never got around a real obstacle in the field.
    strafe_seconds: float = 1.0
    strafe_max_seconds: float = 3.0
    avoid_forward_seconds: float = 1.5
    blocked_backup_sec: float = 1.0          # reverse for clearance per block
    block_retries: int = 3
    # after block_retries side-steps fail, don't give up: back up and re-plan.
    recovery_backup_sec: float = 2.0         # reverse duration per recovery
    max_recovery_cycles: int = 5             # outer bound -> finally FAILED
    # scan recovery: small steps so a tag is not skipped between two stops.
    # 12 x 30 deg = a full 360 deg sweep, stopping often enough to land with
    # a tag inside the camera FOV. After each rotation the robot dwells
    # (stays still) so the detector gets several sharp frames to spot a tag.
    scan_max_rotations: int = 12
    scan_rotate_rad: float = math.pi / 6          # 30 deg per scan pulse
    scan_dwell_sec: float = 2.0                   # still time after each step
    # a blocked:front within this map distance of the goal is read as the
    # goal object itself (arrival), not an obstacle to avoid. Larger than
    # arrived_radius because the ultrasonic trips on the object SURFACE while
    # the goal landmark is its CENTRE.
    blocked_arrival_dist_m: float = 1.0
    # goals with a final heading (tag goals: end up LOOKING at the tag):
    # once inside the arrival radius, turn in place until within this
    # tolerance. Open-loop turns overshoot, so allow a few corrective tries
    # and then arrive regardless rather than hunt forever.
    final_heading_tol_rad: float = 0.3
    final_face_max: int = 3
    # The final heading may only be CONFIRMED by a real tag fix. Dead
    # reckoning assumes every turn was perfect, so it always "confirms" a
    # turn that in reality overshot by 30 deg. Wait this long for a fix
    # before settling for the dead-reckoned heading.
    final_fix_wait_sec: float = 12.0


@dataclass
class NavCommand:
    vx: float
    vy: float
    wz: float
    seconds: float
    kind: str                                    # rotate|forward|strafe|scan_rotate|stop


@dataclass
class NavSession:
    goal_xy: tuple
    goal_id: int | None
    goal_label: str | None
    config: NavConfig = field(default_factory=NavConfig)
    state: State = State.INIT
    block_count: int = 0
    recovery_count: int = 0
    scan_count: int = 0
    queued: list = field(default_factory=list)
    fail_reason: str | None = None
    _recovery_pending: bool = False
    last_tag_t: float | None = None
    _scan_start_t: float | None = None
    _scan_resume_t: float | None = None
    _drive_resume_t: float | None = None     # dwell-until time after a pulse
    # obstacle-vs-target discrimination for a blocked:front event
    _target_ahead: bool = False              # camera sees the goal class close
    # heading to end at (map frame, rad), or None = any heading will do
    final_yaw: float | None = None
    _facing: bool = False                    # position reached, now turning
    _face_count: int = 0
    _final_wait_t: float | None = None       # waiting for a fix to confirm
    _last_goal_dist: float = float('inf')    # map distance at the last tick

    def on_tag_fix(self, t: float):
        """Record that a tag-fix arrived at time ``t``.

        Always clear scan tracking — a fresh tag fix ends any in-progress
        recovery sweep regardless of the current state, so the next stale
        period starts from a clean counter.
        """
        self.last_tag_t = float(t)
        self.scan_count = 0
        self._scan_start_t = None
        self._scan_resume_t = None
        if self.state is State.SCANNING:
            self.state = State.DRIVING

    def on_pi_event(self, event: str):
        if event == 'pulse_done':
            if self.state is State.WAITING_PULSE:
                self.state = State.DRIVING
            elif self.state is State.SCANNING:
                # stay in SCANNING; next tick decides scan-again vs resume
                pass
            # Reset block count only on a clean pulse (no queued avoidance)
            if not self.queued:
                self.block_count = 0
        elif event.startswith('blocked:'):
            side = event.split(':', 1)[1]
            # A front block on the goal itself is an arrival, not an obstacle:
            # treat it as ARRIVED when the camera sees the goal class close
            # ahead, or the map puts us near the goal landmark.
            if side == 'front' and (
                    self._target_ahead
                    or self._last_goal_dist
                    < self.config.blocked_arrival_dist_m):
                self.state = State.ARRIVED
                return
            self._handle_blocked(side)

    def _handle_blocked(self, side: str):
        self.block_count += 1
        if self.block_count >= self.config.block_retries:
            # block_retries side-steps failed — don't give up: queue a
            # back-up + full re-plan instead of failing. Only after
            # max_recovery_cycles of this do we finally give up.
            self.recovery_count += 1
            if self.recovery_count > self.config.max_recovery_cycles:
                self.state = State.FAILED
                self.fail_reason = 'blocked'
                return
            self._recovery_pending = True
            self.block_count = 0
            self.queued.clear()          # drop stale side-steps; recovery wins
            self.state = State.DRIVING
            return
        # Robot frame convention: +vy = strafe LEFT, -vy = strafe RIGHT
        # (matches agent_node._dir_to_twist: 'left' -> (0, +s, 0)).
        # Strafe AWAY from the side that tripped the obstacle sensor.
        if side == 'left':
            vy = -self.config.strafe_speed
        elif side == 'right':
            vy = +self.config.strafe_speed
        else:                                       # 'front' or unknown
            # No side information: try left first, and switch sides after
            # each failed recovery cycle instead of repeating the same one.
            sign = 1.0 if self.recovery_count % 2 == 0 else -1.0
            vy = sign * self.config.strafe_speed
        # Escalate: each consecutive block steps further sideways.
        strafe_sec = min(self.config.strafe_seconds * self.block_count,
                         self.config.strafe_max_seconds)
        # A new block supersedes whatever was left of the previous manoeuvre.
        self.queued.clear()
        # Back up first for clearance, then strafe away, then resume forward.
        # The reverse step matters because the narrow front ultrasonic beam
        # can lose a still-present obstacle after only a small strafe.
        self.queued.append(NavCommand(-self.config.v_max, 0.0, 0.0,
                                      self.config.blocked_backup_sec,
                                      'backward'))
        self.queued.append(NavCommand(0.0, vy, 0.0, strafe_sec, 'strafe'))
        self.queued.append(NavCommand(self.config.v_max, 0.0, 0.0,
                                      self.config.avoid_forward_seconds,
                                      'forward'))
        self.state = State.DRIVING

    def tick(self, robot_pose, now: float, pose_stale: bool = False,
             target_ahead: bool = False, pose_fresh: bool = True):
        """Produce the next motion pulse, or None if nothing to send.

        ``pose_fresh`` — ``robot_pose`` is a tag fix from this very frame
        rather than a dead-reckoned extrapolation.

        ``target_ahead`` — the camera currently sees the goal's object class
        large and centred (i.e. the thing right in front is the target).
        Used to tell an arrival apart from an obstacle on a blocked:front.
        """
        self._target_ahead = target_ahead
        if self.state in (State.ARRIVED, State.FAILED):
            return None
        if self.state is State.WAITING_PULSE:
            return None

        if self.queued:
            cmd = self.queued.pop(0)
            self.state = State.WAITING_PULSE
            return cmd

        # No pose at all (never localized, e.g. NAV started with no tag in
        # view) OR a stale extrapolated pose -> rotate-scan to acquire a tag
        # rather than hang in INIT until the caller times out.
        if robot_pose is None or pose_stale:
            return self._scan_tick(now)

        # Require a CONFIRMED tag fix before driving toward the goal. Until
        # the first real localization of this goto, a dead-reckoned pose is
        # not trusted as a starting point — rotate-scan to acquire a tag.
        if self.last_tag_t is None:
            return self._scan_tick(now)

        # fresh pose -> normal drive
        if self.state is State.SCANNING:
            self.state = State.DRIVING
            self.scan_count = 0
            self._scan_start_t = None
            self._scan_resume_t = None

        x, y, yaw = robot_pose
        dx, dy = self.goal_xy[0] - x, self.goal_xy[1] - y
        dist = math.hypot(dx, dy)
        self._last_goal_dist = dist
        if self._facing or dist < self.config.arrived_radius_m:
            settling = (self._drive_resume_t is not None
                        and now < self._drive_resume_t)
            if self._facing and settling:
                return None             # let the last facing turn settle
            if self.final_yaw is not None:
                err = _wrap(self.final_yaw - yaw)
                turn = (abs(err) > self.config.final_heading_tol_rad
                        and self._face_count < self.config.final_face_max)
                if not turn and not pose_fresh:
                    # About to call the heading good on dead reckoning alone
                    # — hold position and wait for a tag fix to confirm it.
                    self._facing = True
                    if self._final_wait_t is None:
                        self._final_wait_t = now
                    if (now - self._final_wait_t
                            < self.config.final_fix_wait_sec):
                        return None
                if turn:
                    if settling:
                        return None
                    self._final_wait_t = None
                    # Position reached; from here on only turn — never drive
                    # off again because the turn nudged us out of the radius.
                    self._facing = True
                    self._face_count += 1
                    wz = self.config.w_max * (1.0 if err > 0 else -1.0)
                    sec = _clamp(abs(err) / self.config.w_eff,
                                 self.config.min_pulse_sec,
                                 self.config.max_pulse_sec)
                    self._drive_resume_t = (now + sec
                                            + self.config.nav_dwell_sec)
                    self.state = State.WAITING_PULSE
                    return NavCommand(0.0, 0.0, wz, sec, 'rotate')
            self.state = State.ARRIVED
            return NavCommand(0.0, 0.0, 0.0, 0.0, 'stop')

        # recovery: side-steps exhausted -> back up, then re-plan from a
        # fresh pose (the post-pulse dwell gives time to re-localize).
        if self._recovery_pending:
            self._recovery_pending = False
            sec = self.config.recovery_backup_sec
            self._drive_resume_t = now + sec + self.config.nav_dwell_sec
            self.state = State.WAITING_PULSE
            return NavCommand(-self.config.v_max, 0.0, 0.0, sec, 'backward')

        # dwell after the previous pulse: stay still so the detector grabs
        # sharp frames and re-localizes before the next move is computed.
        if self._drive_resume_t is not None and now < self._drive_resume_t:
            return None

        heading_err = _wrap(math.atan2(dy, dx) - yaw)
        if abs(heading_err) > self.config.heading_tol_rad:
            # large error -> dedicated in-place rotate to face the goal.
            # duration uses w_eff (real rate), command uses w_max.
            wz = self.config.w_max * (1.0 if heading_err > 0 else -1.0)
            sec = _clamp(abs(heading_err) / self.config.w_eff,
                         self.config.min_pulse_sec, self.config.max_pulse_sec)
            self._drive_resume_t = now + sec + self.config.nav_dwell_sec
            self.state = State.WAITING_PULSE
            return NavCommand(0.0, 0.0, wz, sec, 'rotate')

        # roughly aimed -> drive forward, blending in proportional steering so
        # the heading self-corrects along the way (curve onto the goal line,
        # no drift-then-snap zigzag).
        sec = _clamp(dist / self.config.v_max,
                     self.config.min_pulse_sec, self.config.max_pulse_sec)
        wz = _clamp(self.config.steer_gain * heading_err,
                    -self.config.forward_steer_max,
                    self.config.forward_steer_max)
        self._drive_resume_t = now + sec + self.config.nav_dwell_sec
        self.state = State.WAITING_PULSE
        return NavCommand(self.config.v_max, 0.0, wz, sec, 'forward')

    def _scan_tick(self, now: float):
        if self.scan_count >= self.config.scan_max_rotations:
            self.state = State.FAILED
            self.fail_reason = 'lost'
            return None
        # Dwell after a scan rotation: stay still until _scan_resume_t so the
        # detector gets sharp, blur-free frames to spot a tag before rotating
        # again. Returning None keeps the robot idle this tick.
        if self._scan_resume_t is not None and now < self._scan_resume_t:
            self.state = State.SCANNING
            return None
        if self._scan_start_t is None:
            self._scan_start_t = now
        self.scan_count += 1
        sec = self.config.scan_rotate_rad / self.config.w_eff
        # next scan step only after this rotation finishes AND the dwell.
        self._scan_resume_t = now + sec + self.config.scan_dwell_sec
        self.state = State.WAITING_PULSE
        return NavCommand(0.0, 0.0, self.config.w_max, sec, 'scan_rotate')


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
