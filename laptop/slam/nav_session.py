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
    heading_tol_rad: float = 0.35
    v_max: float = 0.15
    w_max: float = 0.4
    max_pulse_sec: float = 1.5
    min_pulse_sec: float = 0.2
    strafe_speed: float = 0.15
    strafe_seconds: float = 0.6
    avoid_forward_seconds: float = 0.8
    block_retries: int = 3
    scan_max_rotations: int = 4
    scan_rotate_rad: float = math.pi / 2          # 90 deg per scan pulse


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
    scan_count: int = 0
    queued: list = field(default_factory=list)
    fail_reason: str | None = None
    last_tag_t: float | None = None
    _scan_start_t: float | None = None

    def on_tag_fix(self, t: float):
        """Record that a tag-fix arrived at time ``t``.

        Always clear scan tracking — a fresh tag fix ends any in-progress
        recovery sweep regardless of the current state, so the next stale
        period starts from a clean counter.
        """
        self.last_tag_t = float(t)
        self.scan_count = 0
        self._scan_start_t = None
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
            self._handle_blocked(event.split(':', 1)[1])

    def _handle_blocked(self, side: str):
        self.block_count += 1
        if self.block_count >= self.config.block_retries:
            self.state = State.FAILED
            self.fail_reason = 'blocked'
            return
        vy = (-self.config.strafe_speed if side == 'right'
              else self.config.strafe_speed)
        self.queued.append(NavCommand(0.0, vy, 0.0,
                                      self.config.strafe_seconds, 'strafe'))
        self.queued.append(NavCommand(self.config.v_max, 0.0, 0.0,
                                      self.config.avoid_forward_seconds,
                                      'forward'))
        self.state = State.DRIVING

    def tick(self, robot_pose, now: float, pose_stale: bool = False):
        """Produce the next motion pulse, or None if nothing to send."""
        if self.state in (State.ARRIVED, State.FAILED):
            return None
        if self.state is State.WAITING_PULSE:
            return None
        if robot_pose is None:
            return None

        if self.queued:
            cmd = self.queued.pop(0)
            self.state = State.WAITING_PULSE
            return cmd

        if pose_stale:
            return self._scan_tick(now)

        # fresh pose -> normal drive
        if self.state is State.SCANNING:
            self.state = State.DRIVING
            self.scan_count = 0
            self._scan_start_t = None

        x, y, yaw = robot_pose
        dx, dy = self.goal_xy[0] - x, self.goal_xy[1] - y
        dist = math.hypot(dx, dy)
        if dist < self.config.arrived_radius_m:
            self.state = State.ARRIVED
            return NavCommand(0.0, 0.0, 0.0, 0.0, 'stop')

        heading_err = _wrap(math.atan2(dy, dx) - yaw)
        if abs(heading_err) > self.config.heading_tol_rad:
            wz = self.config.w_max * (1.0 if heading_err > 0 else -1.0)
            sec = _clamp(abs(heading_err) / self.config.w_max,
                         self.config.min_pulse_sec, self.config.max_pulse_sec)
            self.state = State.WAITING_PULSE
            return NavCommand(0.0, 0.0, wz, sec, 'rotate')

        sec = _clamp(dist / self.config.v_max,
                     self.config.min_pulse_sec, self.config.max_pulse_sec)
        self.state = State.WAITING_PULSE
        return NavCommand(self.config.v_max, 0.0, 0.0, sec, 'forward')

    def _scan_tick(self, now: float):
        if self.scan_count >= self.config.scan_max_rotations:
            self.state = State.FAILED
            self.fail_reason = 'lost'
            return None
        if self._scan_start_t is None:
            self._scan_start_t = now
        self.scan_count += 1
        self.state = State.SCANNING
        sec = self.config.scan_rotate_rad / self.config.w_max
        return NavCommand(0.0, 0.0, self.config.w_max, sec, 'scan_rotate')


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
