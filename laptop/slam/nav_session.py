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
    heading_tol_rad: float = 0.15            # >this -> dedicated rotate pulse
    v_max: float = 0.15
    w_max: float = 0.4
    max_pulse_sec: float = 1.5
    min_pulse_sec: float = 0.2
    # forward pulses blend in proportional steering so the robot curves onto
    # the goal line instead of drifting off and snap-correcting later.
    steer_gain: float = 1.5                  # wz per rad of heading error
    forward_steer_max: float = 0.25          # cap on the blended wz (rad/s)
    strafe_speed: float = 0.15
    strafe_seconds: float = 0.6
    avoid_forward_seconds: float = 0.8
    block_retries: int = 3
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
    _scan_resume_t: float | None = None
    # obstacle-vs-target discrimination for a blocked:front event
    _target_ahead: bool = False              # camera sees the goal class close
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
            self.state = State.FAILED
            self.fail_reason = 'blocked'
            return
        # Robot frame convention: +vy = strafe LEFT, -vy = strafe RIGHT
        # (matches agent_node._dir_to_twist: 'left' -> (0, +s, 0)).
        # Strafe AWAY from the side that tripped the obstacle sensor.
        if side == 'left':
            vy = -self.config.strafe_speed
        elif side == 'right':
            vy = +self.config.strafe_speed
        else:                                       # 'front' or unknown
            vy = +self.config.strafe_speed         # default: try left
        self.queued.append(NavCommand(0.0, vy, 0.0,
                                      self.config.strafe_seconds, 'strafe'))
        self.queued.append(NavCommand(self.config.v_max, 0.0, 0.0,
                                      self.config.avoid_forward_seconds,
                                      'forward'))
        self.state = State.DRIVING

    def tick(self, robot_pose, now: float, pose_stale: bool = False,
             target_ahead: bool = False):
        """Produce the next motion pulse, or None if nothing to send.

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
        if dist < self.config.arrived_radius_m:
            self.state = State.ARRIVED
            return NavCommand(0.0, 0.0, 0.0, 0.0, 'stop')

        heading_err = _wrap(math.atan2(dy, dx) - yaw)
        if abs(heading_err) > self.config.heading_tol_rad:
            # large error -> dedicated in-place rotate to face the goal
            wz = self.config.w_max * (1.0 if heading_err > 0 else -1.0)
            sec = _clamp(abs(heading_err) / self.config.w_max,
                         self.config.min_pulse_sec, self.config.max_pulse_sec)
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
        sec = self.config.scan_rotate_rad / self.config.w_max
        # next scan step only after this rotation finishes AND the dwell.
        self._scan_resume_t = now + sec + self.config.scan_dwell_sec
        self.state = State.WAITING_PULSE
        return NavCommand(0.0, 0.0, self.config.w_max, sec, 'scan_rotate')


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
