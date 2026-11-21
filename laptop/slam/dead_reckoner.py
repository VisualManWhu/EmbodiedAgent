"""Bridge tag-fix gaps by integrating issued cmd_vel pulses.

Open-loop on a mecanum kit with motor slip — bounded by a short time and
distance cap. Useful only for traversing the brief intervals between two
in-view tags. Not a substitute for an IMU.
"""
import math
from dataclasses import dataclass


@dataclass
class _Pulse:
    vx: float
    vy: float
    wz: float
    seconds: float
    started_at: float


class DeadReckoner:
    """Replays issued pulses from the last tag fix to estimate pose."""

    def __init__(self, time_limit_sec: float = 5.0,
                 distance_limit_m: float = 0.5,
                 integration_dt: float = 0.05):
        self.time_limit_sec = float(time_limit_sec)
        self.distance_limit_m = float(distance_limit_m)
        self.integration_dt = float(integration_dt)
        self.last_pose = None
        self.last_t = None
        self.pulses: list[_Pulse] = []

    def reset(self, pose, t: float):
        """Snap the reckoner to a known pose at time ``t``.

        Motion already executed before ``t`` is baked into the new pose, so
        it is dropped — but the part of a pulse that runs on AFTER ``t`` is
        kept. A tag fix routinely lands just after a pulse is issued (the tag
        is still in view while the motors ramp up); discarding that pulse
        made the reckoner report the pre-move pose once the tag left the
        frame, and nav re-issued the same rotation.
        """
        t = float(t)
        self.last_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        self.last_t = t
        remaining = []
        for p in self.pulses:
            end = p.started_at + p.seconds
            if end <= t:
                continue                            # fully executed already
            start = max(p.started_at, t)
            remaining.append(_Pulse(p.vx, p.vy, p.wz, end - start, start))
        self.pulses = remaining

    def record_pulse(self, vx: float, vy: float, wz: float,
                     seconds: float, started_at: float):
        """Append a pulse to the integration buffer."""
        self.pulses.append(_Pulse(float(vx), float(vy), float(wz),
                                  float(seconds), float(started_at)))

    def truncate_pulses(self, now: float):
        """The robot was stopped early (the front sensor tripped): pulses
        still running end NOW, not at their planned duration. Without this
        the pose was carried on to the end of the pulse — once to within
        7 cm of a goal the robot was still half a metre from."""
        kept = []
        for p in self.pulses:
            if p.started_at >= now:
                continue                            # never started
            end = p.started_at + p.seconds
            if end > now:
                p = _Pulse(p.vx, p.vy, p.wz, now - p.started_at, p.started_at)
            kept.append(p)
        self.pulses = kept

    def cancel_pulse(self, started_at: float):
        """Forget a recorded pulse that turned out not to have moved the
        robot (it was stuck), so the pose is not carried on without it."""
        self.pulses = [p for p in self.pulses
                       if abs(p.started_at - started_at) > 1e-6]

    def pose_at(self, now: float):
        """``(pose, stale)`` — pose extrapolated to ``now`` if within caps."""
        if self.last_pose is None or self.last_t is None:
            return None, True
        if (now - self.last_t) > self.time_limit_sec:
            return self.last_pose, True

        x, y, yaw = self.last_pose
        for p in self.pulses:
            elapsed = min(p.seconds, max(0.0, now - p.started_at))
            if elapsed <= 0.0:
                continue
            steps = max(1, int(math.ceil(elapsed / self.integration_dt)))
            dt = elapsed / steps
            for _ in range(steps):
                mid_yaw = yaw + 0.5 * p.wz * dt
                dx_w = p.vx * math.cos(mid_yaw) - p.vy * math.sin(mid_yaw)
                dy_w = p.vx * math.sin(mid_yaw) + p.vy * math.cos(mid_yaw)
                x += dx_w * dt
                y += dy_w * dt
                yaw += p.wz * dt

        x0, y0, _ = self.last_pose
        stale = math.hypot(x - x0, y - y0) > self.distance_limit_m
        return (x, y, yaw), stale


def modeled_yaw_rate(vx: float, vy: float, wz: float,
                     w_cmd_ref: float, w_eff: float,
                     wheel_half_sum_m: float = 0.165,
                     speed_gain: float = 200.0,
                     min_pct: float = 30.0, max_pct: float = 45.0,
                     move_eps: float = 0.01) -> float:
    """Yaw rate (rad/s) the robot actually achieves for a commanded twist.

    Mirrors ``motor_node``: mecanum mixing -> wheel_norm * speed_gain ->
    duty percent clamped to [min_pct, max_pct] (the floor exists to overcome
    motor stall). The clamp makes the response very non-linear: an in-place
    turn commanded at 0.5 rad/s runs every wheel at the floor and really
    turns at ~1.12 rad/s, while the same wz blended into a forward pulse only
    produces the small left/right duty DIFFERENCE left after the floor.

    Calibrated against the one hand-measured point: an in-place turn at
    ``w_cmd_ref`` yields ``w_eff``. Yaw rate is taken as proportional to the
    right-minus-left duty difference. The defaults mirror the Pi's
    params.yaml (wheel_base_lx + ly, speed_gain, min/max_move_percent).
    """
    def duty(w):
        if abs(w) < move_eps:
            return 0.0
        pct = max(min_pct, min(max_pct, abs(w) * speed_gain))
        return pct if w > 0 else -pct

    k = wz * wheel_half_sum_m
    fl, fr = vx - vy - k, vx + vy + k
    rl, rr = vx + vy - k, vx - vy + k
    diff = (duty(fr) + duty(rr) - duty(fl) - duty(rl)) / 4.0
    ref = duty(abs(w_cmd_ref) * wheel_half_sum_m)       # in-place duty
    if ref == 0.0:
        return 0.0
    return w_eff * diff / ref
