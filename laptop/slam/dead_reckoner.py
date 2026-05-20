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
        """Snap the reckoner to a known pose and clear the pulse buffer."""
        self.last_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        self.last_t = float(t)
        self.pulses.clear()

    def record_pulse(self, vx: float, vy: float, wz: float,
                     seconds: float, started_at: float):
        """Append a pulse to the integration buffer."""
        self.pulses.append(_Pulse(float(vx), float(vy), float(wz),
                                  float(seconds), float(started_at)))

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
