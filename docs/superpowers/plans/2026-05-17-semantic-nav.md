# Semantic Navigation P2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map-aware navigation that drives the robot to an AprilTag-frame goal — by landmark id, by class (with Telegram disambiguation), via patrol/room-tour, or as a fallback to a timed-out SEARCH.

**Architecture:** High-level state machine on the laptop produces motion pulses; Pi `agent_node` gains a thin `NAV` mode that executes one pulse and reports `pulse_done` / `blocked:<side>`. Laptop bridges short tag-fix gaps with `cmd_vel` dead-reckoning, recovers by rotating to re-acquire a tag, and orchestrates Telegram inline-button disambiguation, SEARCH→NAV chain, and patrol sequencing.

**Tech Stack:** Python 3.11 (laptop), ROS2 Humble + Python (Pi), `requests`, `numpy`, `python-telegram-bot`, existing `slam/` package.

**Reference spec:** [`docs/superpowers/specs/2026-05-17-semantic-nav-design.md`](../specs/2026-05-17-semantic-nav-design.md).

---

## File map

**New (laptop):**
- `laptop/slam/landmark_selector.py` — class → ranked candidate landmarks.
- `laptop/slam/dead_reckoner.py` — integrates motion pulses to extrapolate pose between tag fixes.
- `laptop/slam/nav_session.py` — single-goto state machine.
- `laptop/tests/test_landmark_selector.py`
- `laptop/tests/test_dead_reckoner.py`
- `laptop/tests/test_nav_session.py`

**Modified (laptop):**
- `laptop/nl_parser.py` — new actions (`goto`, `patrol`, `room_tour`).
- `laptop/laptop_detector.py` — wire NavSession into main loop, add CLI flags, poll Pi `/status` for events.
- `laptop/telegram_bot.py` — inline-keyboard disambiguation, chain controller, patrol orchestration.
- `laptop/tests/test_nl_parser.py`

**Modified (Pi):**
- `src/embodied_mvp/embodied_mvp/agent_node.py` — new `NAV` mode + `nav_pulse` / `nav_stop` / `scan` actions, blocked-event posting.

---

## Task 1: `landmark_selector`

**Files:**
- Create: `laptop/slam/landmark_selector.py`
- Test: `laptop/tests/test_landmark_selector.py`

- [ ] **Step 1: Write the failing test**

```python
# laptop/tests/test_landmark_selector.py
"""Unit tests for landmark_selector: class -> ranked candidates."""
import numpy as np
import pytest

from slam.landmark_selector import (by_class, by_id, by_tag,
                                    nearest_by_class)
from slam.semantic_map import CONFIRMED, TENTATIVE, Landmark, SemanticMap


def _add(smap, lid, label, x, y, conf=1.0, state=CONFIRMED):
    lm = Landmark(id=lid)
    lm.class_votes = {label: conf}
    lm.position = np.array([x, y, 0.0])
    lm.state = state
    smap.landmarks[lid] = lm
    return lm


def test_by_id_returns_landmark():
    smap = SemanticMap()
    _add(smap, 1, 'chair', 1.0, 0.0)
    assert by_id(smap, 1).id == 1
    assert by_id(smap, 99) is None


def test_by_class_filters_state_and_confidence():
    smap = SemanticMap()
    _add(smap, 1, 'chair', 1.0, 0.0, conf=0.9)
    _add(smap, 2, 'chair', 2.0, 0.0, conf=0.5)              # below floor
    _add(smap, 3, 'chair', 3.0, 0.0, conf=0.9, state=TENTATIVE)
    _add(smap, 4, 'couch', 1.0, 1.0, conf=0.9)              # wrong class
    out = by_class(smap, 'chair', robot_xy=(0.0, 0.0),
                   conf_min=0.7, state=CONFIRMED)
    assert [lm.id for lm in out] == [1]


def test_by_class_orders_by_distance_then_id():
    smap = SemanticMap()
    _add(smap, 5, 'chair', 3.0, 0.0)
    _add(smap, 6, 'chair', 1.0, 0.0)
    _add(smap, 7, 'chair', 1.0, 0.0)                         # same dist as 6
    out = by_class(smap, 'chair', robot_xy=(0.0, 0.0))
    assert [lm.id for lm in out] == [6, 7, 5]


def test_nearest_by_class_returns_first_or_none():
    smap = SemanticMap()
    assert nearest_by_class(smap, 'chair', (0.0, 0.0)) is None
    _add(smap, 1, 'chair', 1.0, 0.0)
    assert nearest_by_class(smap, 'chair', (0.0, 0.0)).id == 1


def test_by_tag_returns_xy_or_none():
    tag_map = {0: {'x': 0.0, 'y': 0.0, 'z': 0.3, 'yaw_deg': 0.0, 'size_m': 0.1}}
    assert by_tag(tag_map, 0) == (0.0, 0.0)
    assert by_tag(tag_map, 9) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd laptop && python -m pytest tests/test_landmark_selector.py -v`
Expected: ImportError — `slam.landmark_selector` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# laptop/slam/landmark_selector.py
"""Pick which landmark a class-level goto refers to.

Pure functions over ``SemanticMap`` — no side effects, no state.
"""
import math

from .semantic_map import CONFIRMED


def by_id(smap, lid):
    """Look up a landmark by id, or return None."""
    return smap.landmarks.get(int(lid))


def by_class(smap, cls, robot_xy, conf_min: float = 0.7,
             state: str | None = CONFIRMED):
    """Landmarks matching ``cls``, filtered by state/confidence, nearest first.

    Ties on distance break deterministically by ascending id.
    """
    candidates = []
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    for lm in smap.landmarks.values():
        if state is not None and lm.state != state:
            continue
        if lm.label != cls:
            continue
        if lm.confidence < conf_min:
            continue
        d = math.hypot(lm.position[0] - rx, lm.position[1] - ry)
        candidates.append((d, lm.id, lm))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return [lm for _, _, lm in candidates]


def nearest_by_class(smap, cls, robot_xy, **kw):
    """Single nearest matching landmark, or None."""
    cands = by_class(smap, cls, robot_xy, **kw)
    return cands[0] if cands else None


def by_tag(tag_map, tag_id):
    """``(x, y)`` of an AprilTag from the tag map, or None."""
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    return (float(e['x']), float(e['y']))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd laptop && python -m pytest tests/test_landmark_selector.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add laptop/slam/landmark_selector.py laptop/tests/test_landmark_selector.py
git commit -m "feat(slam): landmark_selector — class -> ranked candidates"
```

---

## Task 2: `dead_reckoner`

**Files:**
- Create: `laptop/slam/dead_reckoner.py`
- Test: `laptop/tests/test_dead_reckoner.py`

- [ ] **Step 1: Write the failing test**

```python
# laptop/tests/test_dead_reckoner.py
"""Unit tests for dead_reckoner: integrate cmd_vel pulses between tag fixes."""
import math

import pytest

from slam.dead_reckoner import DeadReckoner


def test_no_pose_yet_returns_none():
    dr = DeadReckoner()
    pose, stale = dr.pose_at(now=10.0)
    assert pose is None
    assert stale is True


def test_forward_pulse_translates_along_yaw():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.15, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    pose, stale = dr.pose_at(now=1.0)
    x, y, yaw = pose
    assert x == pytest.approx(0.15, abs=1e-3)
    assert y == pytest.approx(0.0, abs=1e-3)
    assert yaw == pytest.approx(0.0, abs=1e-3)
    assert stale is False


def test_pure_rotation():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.0, wz=1.0, seconds=1.0, started_at=0.0)
    pose, _ = dr.pose_at(now=1.0)
    _, _, yaw = pose
    assert yaw == pytest.approx(1.0, abs=1e-3)


def test_strafe_pulse_translates_sideways():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.0, vy=0.15, wz=0.0, seconds=1.0, started_at=0.0)
    pose, _ = dr.pose_at(now=1.0)
    x, y, _ = pose
    assert x == pytest.approx(0.0, abs=1e-3)
    assert y == pytest.approx(0.15, abs=1e-3)


def test_partial_pulse_only_counts_elapsed_portion():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.2, vy=0.0, wz=0.0, seconds=2.0, started_at=0.0)
    pose, _ = dr.pose_at(now=0.5)         # only 0.5 s elapsed
    x, _, _ = pose
    assert x == pytest.approx(0.10, abs=5e-3)


def test_time_cap_marks_stale():
    dr = DeadReckoner(time_limit_sec=1.0)
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.05, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    _, stale = dr.pose_at(now=5.0)
    assert stale is True


def test_distance_cap_marks_stale():
    dr = DeadReckoner(distance_limit_m=0.1)
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.5, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    _, stale = dr.pose_at(now=1.0)
    assert stale is True


def test_reset_clears_pulses():
    dr = DeadReckoner()
    dr.reset((0.0, 0.0, 0.0), t=0.0)
    dr.record_pulse(vx=0.15, vy=0.0, wz=0.0, seconds=1.0, started_at=0.0)
    dr.reset((1.0, 2.0, 0.5), t=10.0)
    pose, stale = dr.pose_at(now=10.0)
    assert pose == (1.0, 2.0, 0.5)
    assert stale is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd laptop && python -m pytest tests/test_dead_reckoner.py -v`
Expected: ImportError — `slam.dead_reckoner` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# laptop/slam/dead_reckoner.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd laptop && python -m pytest tests/test_dead_reckoner.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add laptop/slam/dead_reckoner.py laptop/tests/test_dead_reckoner.py
git commit -m "feat(slam): dead_reckoner — bridge tag-fix gaps by pulse integration"
```

---

## Task 3: `nav_session` core state machine

**Files:**
- Create: `laptop/slam/nav_session.py`
- Test: `laptop/tests/test_nav_session.py`

- [ ] **Step 1: Write the failing test**

```python
# laptop/tests/test_nav_session.py
"""Unit tests for NavSession: pose+goal -> motion pulses + Pi event handling."""
import math

import pytest

from slam.nav_session import NavCommand, NavConfig, NavSession, State


def _sess(goal=(2.0, 0.0), **cfg):
    return NavSession(goal_xy=goal, goal_id=1, goal_label='chair',
                      config=NavConfig(**cfg))


def test_arrives_when_inside_radius():
    s = _sess(arrived_radius_m=0.4)
    cmd = s.tick(robot_pose=(2.0, 0.0, 0.0), now=0.0)
    assert s.state is State.ARRIVED
    assert cmd.kind == 'stop'


def test_rotates_when_heading_off():
    s = _sess(heading_tol_rad=0.35)
    cmd = s.tick(robot_pose=(0.0, 0.0, math.pi / 2), now=0.0)  # goal to the right
    assert cmd.kind == 'rotate'
    assert cmd.wz < 0                                          # turn right
    assert cmd.vx == 0.0


def test_drives_forward_when_heading_ok():
    s = _sess()
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.0), now=0.0)
    assert cmd.kind == 'forward'
    assert cmd.vx > 0
    assert cmd.wz == 0.0


def test_no_command_while_waiting_for_pulse():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)                                # one pulse issued
    assert s.tick((0.0, 0.0, 0.0), 0.1) is None                 # no overlap


def test_pulse_done_unblocks_next_tick():
    s = _sess()
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('pulse_done')
    cmd = s.tick((0.05, 0.0, 0.0), 0.5)
    assert cmd is not None
    assert cmd.kind == 'forward'


def test_blocked_queues_strafe_then_forward():
    s = _sess(strafe_speed=0.15)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:left')
    cmd1 = s.tick((0.0, 0.0, 0.0), 0.1)
    cmd2 = s.tick((0.0, 0.0, 0.0), 0.2)                         # next after pulse_done
    s.on_pi_event('pulse_done')
    cmd2 = s.tick((0.0, 0.0, 0.0), 0.3)
    assert cmd1.kind == 'strafe'
    assert cmd1.vy != 0
    assert cmd2.kind == 'forward'


def test_three_blocks_fail():
    s = _sess(block_retries=3)
    for _ in range(3):
        s.tick((0.0, 0.0, 0.0), 0.0)
        s.on_pi_event('blocked:front')
    assert s.state is State.FAILED
    assert s.fail_reason == 'blocked'


def test_block_count_resets_on_clean_pulse():
    s = _sess(block_retries=3)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('blocked:front')
    s.tick((0.0, 0.0, 0.0), 0.1)                                # strafe queued
    s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 0.2)                                # forward of strafe seq
    s.on_pi_event('pulse_done')
    assert s.block_count == 0


def test_scan_when_pose_stale_and_no_tag():
    s = _sess(scan_max_rotations=4)
    s.on_tag_fix(t=0.0)
    s.tick((0.0, 0.0, 0.0), 0.0)
    s.on_pi_event('pulse_done')
    # 10 s later, no further tag fix -> trigger scan
    cmd = s.tick(robot_pose=(0.0, 0.0, 0.0), now=10.0, pose_stale=True)
    assert cmd.kind == 'scan_rotate'


def test_scan_exhausted_fails():
    s = _sess(scan_max_rotations=2)
    s.on_tag_fix(t=0.0)
    for i in range(2):
        s.tick((0.0, 0.0, 0.0), 10.0 + i, pose_stale=True)
        s.on_pi_event('pulse_done')
    s.tick((0.0, 0.0, 0.0), 20.0, pose_stale=True)
    assert s.state is State.FAILED
    assert s.fail_reason == 'lost'


def test_tag_fix_during_scan_resumes_drive():
    s = _sess(scan_max_rotations=4)
    s.on_tag_fix(t=0.0)
    s.tick((0.0, 0.0, 0.0), 10.0, pose_stale=True)               # scan pulse
    s.on_pi_event('pulse_done')
    s.on_tag_fix(t=11.0)                                         # recovered
    cmd = s.tick((0.1, 0.0, 0.0), 11.5, pose_stale=False)
    assert cmd.kind in ('rotate', 'forward')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd laptop && python -m pytest tests/test_nav_session.py -v`
Expected: ImportError — `slam.nav_session` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# laptop/slam/nav_session.py
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
        """Record that a tag-fix arrived at time ``t``."""
        self.last_tag_t = float(t)
        if self.state is State.SCANNING:
            # interrupted scan, resume normal driving next tick
            self.state = State.DRIVING
            self.scan_count = 0
            self._scan_start_t = None

    def on_pi_event(self, event: str):
        if event == 'pulse_done':
            if self.state is State.WAITING_PULSE:
                self.state = State.DRIVING if not self.queued else State.WAITING_PULSE
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
        if self._scan_start_t is None:
            self._scan_start_t = now
        if self.scan_count >= self.config.scan_max_rotations:
            self.state = State.FAILED
            self.fail_reason = 'lost'
            return None
        self.scan_count += 1
        self.state = State.WAITING_PULSE
        sec = self.config.scan_rotate_rad / self.config.w_max
        return NavCommand(0.0, 0.0, self.config.w_max, sec, 'scan_rotate')


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd laptop && python -m pytest tests/test_nav_session.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add laptop/slam/nav_session.py laptop/tests/test_nav_session.py
git commit -m "feat(slam): NavSession — single-goto state machine"
```

---

## Task 4: Pi `agent_node` NAV mode + `nav_pulse` action

**Files:**
- Modify: `src/embodied_mvp/embodied_mvp/agent_node.py`

- [ ] **Step 1: Read the existing `apply_command` and `tick` methods**

Read `agent_node.py` to confirm where MANUAL pulse handling lives (currently `tick_manual` consumes `manual_twist` + `manual_deadline`). NAV pulse mirrors that but also accepts `vy` and posts events.

- [ ] **Step 2: Modify `apply_command` to accept `nav_pulse` / `nav_stop`**

In `agent_node.py`, replace the `apply_command` body to add two new actions. Locate the existing block:

```python
def apply_command(self, cmd, now):
    action = cmd.get('action')
    if action == 'stop':
        self.mode = 'IDLE'
    elif action == 'move':
        ...
```

Insert before the trailing `elif action == 'rotate_photo':`:

```python
        elif action == 'nav_pulse':
            self.mode = 'NAV'
            vx = float(cmd.get('vx', 0.0))
            vy = float(cmd.get('vy', 0.0))
            wz = float(cmd.get('wz', 0.0))
            sec = float(cmd.get('seconds', 0.5))
            self.nav_twist = (vx, vy, wz)
            self.nav_deadline = now + max(0.05, min(2.0, sec))
            self.nav_blocked = False
        elif action == 'nav_stop':
            self.mode = 'IDLE'
            self.nav_twist = (0.0, 0.0, 0.0)
            self.nav_deadline = 0.0
```

- [ ] **Step 3: Initialize NAV state in `__init__`**

Find the `self.rp_index = 0` block (rotate_photo state) and add right after it:

```python
        self.nav_twist = (0.0, 0.0, 0.0)
        self.nav_deadline = 0.0
        self.nav_blocked = False
```

- [ ] **Step 4: Add `tick_nav`**

Before `def tick(self):`, insert:

```python
    def tick_nav(self, now):
        """Execute a single NAV pulse. Abort on ultrasonic / side IR; report
        events back to the laptop via the command_server status channel.
        """
        if now >= self.nav_deadline:
            self.publish_full_cmd(0.0, 0.0, 0.0)
            if not self.nav_blocked:
                self.cmd_server.post_event('pulse_done')
            self.mode = 'IDLE'
            return
        # obstacle checks — abort the pulse, post a typed event
        if self.ir_range < self.stop_d:
            self.publish_full_cmd(0.0, 0.0, 0.0)
            self.cmd_server.post_event('blocked:front')
            self.nav_blocked = True
            self.mode = 'IDLE'
            return
        if self.side_ir_enabled and self.obs_left:
            self.publish_full_cmd(0.0, 0.0, 0.0)
            self.cmd_server.post_event('blocked:left')
            self.nav_blocked = True
            self.mode = 'IDLE'
            return
        if self.side_ir_enabled and self.obs_right:
            self.publish_full_cmd(0.0, 0.0, 0.0)
            self.cmd_server.post_event('blocked:right')
            self.nav_blocked = True
            self.mode = 'IDLE'
            return
        vx, vy, wz = self.nav_twist
        self.publish_full_cmd(vx, vy, wz)
```

- [ ] **Step 5: Route `NAV` mode in `tick`**

Find `elif self.mode == 'ROTATE_PHOTO':` in `tick` and insert before it:

```python
        elif self.mode == 'NAV':
            self.tick_nav(now)
```

- [ ] **Step 6: Syntax check + commit**

```bash
python -m py_compile src/embodied_mvp/embodied_mvp/agent_node.py
git add src/embodied_mvp/embodied_mvp/agent_node.py
git commit -m "feat(agent): NAV mode + nav_pulse / nav_stop with blocked-event posting"
```

Expected: `py_compile` succeeds silently.

---

## Task 5: Integrate `NavSession` into `laptop_detector.py`

**Files:**
- Modify: `laptop/laptop_detector.py`

- [ ] **Step 1: Add nav CLI flags + import the new module**

In `laptop_detector.py` near the top, add to the imports block (after `from ultralytics import YOLO`):

```python
import math
```

In `main()`, after the existing fusion CLI flags (`--miss-prune`), add:

```python
    # navigation
    ap.add_argument('--arrived-radius-m', type=float, default=0.4)
    ap.add_argument('--nav-v-max', type=float, default=0.15)
    ap.add_argument('--nav-w-max', type=float, default=0.4)
    ap.add_argument('--max-pulse-sec', type=float, default=1.5)
    ap.add_argument('--no-tag-grace-sec', type=float, default=5.0)
    ap.add_argument('--dr-distance-limit-m', type=float, default=0.5)
    ap.add_argument('--dr-time-limit-sec', type=float, default=5.0)
    ap.add_argument('--block-retries', type=int, default=3)
    ap.add_argument('--scan-max-rotations', type=int, default=4)
    ap.add_argument('--landmark-conf-min', type=float, default=0.7)
```

- [ ] **Step 2: Extend `SlamRunner` with nav glue**

In `SlamRunner.__init__`, after the existing renderer setup, add (use lazy imports so the new modules don't load when `--slam` is off):

```python
        from slam.dead_reckoner import DeadReckoner
        from slam.nav_session import NavConfig

        self.dead_reckoner = DeadReckoner(
            time_limit_sec=fusion_overrides.get('dr_time_limit_sec', 5.0)
            if fusion_overrides else 5.0,
            distance_limit_m=fusion_overrides.get('dr_distance_limit_m', 0.5)
            if fusion_overrides else 0.5,
        )
        self.nav_config = NavConfig()           # caller may overwrite fields
        self.session = None                     # NavSession | None
        self.event_url = f'http://{pi_ip}:9091/status'
        self.cmd_url = f'http://{pi_ip}:9091/command'
```

(Keep the existing `self.post_url` / `self.session` request session intact; rename request session to `self.http` to avoid colliding with the NavSession. Replace every `self.session.post` and `self.session.get` in this class with `self.http.post` / `self.http.get`.)

- [ ] **Step 3: Add nav lifecycle methods to `SlamRunner`**

Inside `SlamRunner`, add:

```python
    def start_goto(self, goal_xy, goal_id, goal_label):
        from slam.nav_session import NavSession
        self.session = NavSession(goal_xy=goal_xy, goal_id=goal_id,
                                  goal_label=goal_label,
                                  config=self.nav_config)
        print(f'NAV start -> {goal_label} id={goal_id} at {goal_xy}')

    def stop_goto(self):
        if self.session is not None:
            print('NAV stop')
            try:
                self.http.post(self.cmd_url, json={'action': 'nav_stop'},
                               timeout=0.5)
            except Exception:                  # noqa: BLE001
                pass
            self.session = None
```

- [ ] **Step 4: Tick the nav inside `process`**

Replace `SlamRunner.process` to add nav update after SLAM fuse:

Locate the existing block ending with `self.last_robot = (rx, ry, ryaw)` inside the `if located:` branch. Right below that branch (still inside `process`), add:

```python
        # --- nav update -------------------------------------------------
        nav_status = None
        if self.session is not None:
            if located:
                rx, ry, ryaw = pose['base_xytheta']
                self.dead_reckoner.reset((rx, ry, ryaw), now)
                self.session.on_tag_fix(now)
                robot_pose, pose_stale = (rx, ry, ryaw), False
            else:
                robot_pose, pose_stale = self.dead_reckoner.pose_at(now)
            cmd = self.session.tick(robot_pose, now, pose_stale=pose_stale)
            if cmd is not None and cmd.kind != 'stop':
                self._send_nav(cmd, now)
            self._drain_pi_events()
            nav_status = self.session.state.value
        return (len(tag_dets), located, nav_status)
```

Replace the existing single `return len(tag_dets), located` line at the bottom of `process` with the new three-element return above. If `self.session is None`, `nav_status` is the default `None` set at the top of this added block — keeping the tuple width consistent.

- [ ] **Step 5: Pulse + event-drain helpers**

Add to `SlamRunner`:

```python
    def _send_nav(self, cmd, now):
        payload = {'action': 'nav_pulse',
                   'vx': cmd.vx, 'vy': cmd.vy, 'wz': cmd.wz,
                   'seconds': cmd.seconds}
        try:
            self.http.post(self.cmd_url, json=payload, timeout=0.5)
        except Exception:                       # noqa: BLE001
            return
        self.dead_reckoner.record_pulse(cmd.vx, cmd.vy, cmd.wz,
                                        cmd.seconds, now)

    def _drain_pi_events(self):
        try:
            r = self.http.get(self.event_url, timeout=0.5)
            if r.status_code != 200:
                return
            event = r.json().get('event') or ''
        except Exception:                       # noqa: BLE001
            return
        if event:
            self.session.on_pi_event(event)
            if self.session.state.value in ('ARRIVED', 'FAILED'):
                self._publish_done()

    def _publish_done(self):
        reason = (self.session.fail_reason
                  if self.session.state.value == 'FAILED' else None)
        result = {'event': 'nav_done',
                  'state': self.session.state.value,
                  'goal_label': self.session.goal_label,
                  'goal_id': self.session.goal_id,
                  'reason': reason}
        # bot consumes via a dedicated HTTP endpoint on the laptop; for the
        # first cut, just print — Task 7 wires up the bot relay.
        print(f'NAV result: {result}')
        if self.session.state.value == 'FAILED':
            try:
                self.http.post(self.cmd_url,
                               json={'action': 'nav_stop'}, timeout=0.5)
            except Exception:
                pass
        self.session = None
```

- [ ] **Step 6: Adjust the main-loop unpack of `slam.process`**

Find in `main()`:

```python
                if slam is not None:
                    try:
                        n_tags, located = slam.process(frame, dets, pan_yaw,
                                                       pan_tilt, time.time())
                        slam_info = ...
```

Replace with:

```python
                if slam is not None:
                    try:
                        n_tags, located, nav_state = slam.process(
                            frame, dets, pan_yaw, pan_tilt, time.time())
                        slam_info = (f'  tags={n_tags} '
                                     f'{"LOCATED" if located else "no-fix"}'
                                     f'  nav={nav_state or "-"}')
                    except Exception as e:      # noqa: BLE001
                        slam_info = f'  slam-error: {e}'
```

- [ ] **Step 7: Syntax check + manual smoke**

```bash
python -m py_compile laptop/laptop_detector.py
cd laptop
python -c "from laptop_detector import SlamRunner; print('ok')"
```

Expected: prints `ok`. No exception. (Full integration smoke needs the bot — covered in Task 7.)

- [ ] **Step 8: Commit**

```bash
git add laptop/laptop_detector.py
git commit -m "feat(detector): wire NavSession + dead_reckoner into the main loop"
```

---

## Task 6: `nl_parser` — new actions

**Files:**
- Modify: `laptop/nl_parser.py`
- Modify: `laptop/tests/test_nl_parser.py`

- [ ] **Step 1: Write the failing tests**

Append to `laptop/tests/test_nl_parser.py`:

```python
def test_goto_by_id():
    assert parse('去 id 3') == {'action': 'goto', 'landmark_id': 3}
    assert parse('goto id 7') == {'action': 'goto', 'landmark_id': 7}


def test_goto_origin():
    assert parse('回原点') == {'action': 'goto', 'origin': True}
    assert parse('回到原点') == {'action': 'goto', 'origin': True}


def test_goto_class():
    assert parse('去椅子') == {'action': 'goto', 'target_class': 'chair'}
    assert parse('goto chair') == {'action': 'goto', 'target_class': 'chair'}


def test_patrol():
    assert parse('巡逻') == {'action': 'patrol'}


def test_room_tour():
    assert parse('绕室') == {'action': 'room_tour'}
    assert parse('绕室一周') == {'action': 'room_tour'}


def test_chain_search_still_routes_to_find():
    # 'find <class>' / '去找<class>' continue to mean SEARCH; chain logic is
    # in telegram_bot.py, not here.
    assert parse('去找椅子') == {'action': 'find', 'target': 'chair'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd laptop && python -m pytest tests/test_nl_parser.py -v`
Expected: 6 new failures.

- [ ] **Step 3: Implement parser changes**

In `nl_parser.py`, modify the docstring's command list comment to mention the new actions (a single edit; see the file for placement). Then in `parse()`, insert these branches **before** the existing `if any(k in t for k in ('找', '寻找', '靠近', 'find')):` line:

```python
    # goto id N -- must match before plain '左/前/后' direction matchers
    import re as _re
    m = _re.search(r'(?:去|goto)\s*id\s*(\d+)', t, flags=_re.IGNORECASE)
    if m:
        return {'action': 'goto', 'landmark_id': int(m.group(1))}

    if any(k in t for k in ('回原点', '回到原点', 'go home', 'home')):
        return {'action': 'goto', 'origin': True}

    if any(k in t for k in ('巡逻', 'patrol')):
        return {'action': 'patrol'}

    if any(k in t for k in ('绕室', 'room tour')):
        return {'action': 'room_tour'}

    # 'goto <class>' / '去<class>' (but not '去找') -- requires a target match
    if '去找' not in t and ('去' in t or t.lower().startswith('goto')):
        target = _match_target(t)
        if target:
            return {'action': 'goto', 'target_class': target}
```

(Hoist the `import re` to the top of the file if not already present.)

- [ ] **Step 4: Run tests to verify pass**

Run: `cd laptop && python -m pytest tests/test_nl_parser.py -v`
Expected: all pass (existing + 6 new).

- [ ] **Step 5: Commit**

```bash
git add laptop/nl_parser.py laptop/tests/test_nl_parser.py
git commit -m "feat(parser): goto by id/class/origin, patrol, room_tour"
```

---

## Task 7: Bot inline-keyboard disambiguation + nav lifecycle

**Files:**
- Modify: `laptop/telegram_bot.py`

- [ ] **Step 1: Add candidate query + goto endpoints on the laptop detector**

The bot needs to ask the detector "what candidates exist?" and "please start a goto." Add a small HTTP server to `SlamRunner` so the bot can request these without imports.

In `laptop/laptop_detector.py`, add to `SlamRunner.__init__` (right after `self.map_server = MapImageServer(map_port)`):

```python
        self.nav_api_port = map_port + 1            # default 8092
        self._nav_done_lock = threading.Lock()
        self._nav_done_event = None
        self._start_nav_api()
```

Add methods (still inside `SlamRunner`):

```python
    def _start_nav_api(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import json as _json
        runner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/nav/done':
                    with runner._nav_done_lock:
                        ev = runner._nav_done_event
                        runner._nav_done_event = None
                    body = _json.dumps(ev or {}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/nav/candidates':
                    runner._handle_candidates(self)
                elif self.path == '/nav/goto':
                    runner._handle_goto(self)
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer(('127.0.0.1', self.nav_api_port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'nav api at http://127.0.0.1:{self.nav_api_port}/')

    def _handle_candidates(self, h):
        import json as _json
        from slam.landmark_selector import by_class
        n = int(h.headers.get('Content-Length', 0))
        body = _json.loads(h.rfile.read(n))
        cls = body['target_class']
        robot = self.last_robot or (0.0, 0.0, 0.0)
        cands = by_class(self.smap, cls, (robot[0], robot[1]),
                         conf_min=0.7)
        out = [{'id': lm.id, 'label': lm.label,
                'x': float(lm.position[0]), 'y': float(lm.position[1]),
                'dist_m': float(((lm.position[0]-robot[0])**2
                                 + (lm.position[1]-robot[1])**2) ** 0.5)}
               for lm in cands]
        body = _json.dumps({'candidates': out}).encode()
        h.send_response(200); h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(body))); h.end_headers()
        h.wfile.write(body)

    def _handle_goto(self, h):
        import json as _json
        from slam.landmark_selector import by_id, by_tag
        n = int(h.headers.get('Content-Length', 0))
        req = _json.loads(h.rfile.read(n))
        if 'landmark_id' in req:
            lm = by_id(self.smap, req['landmark_id'])
            if lm is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto((float(lm.position[0]), float(lm.position[1])),
                            lm.id, lm.label)
        elif 'tag_id' in req:
            xy = by_tag(self.tag_map, req['tag_id'])
            if xy is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto(xy, None, f"tag{req['tag_id']}")
        elif 'xy' in req:
            self.start_goto(tuple(req['xy']), None, req.get('label', 'point'))
        else:
            h.send_response(400); h.end_headers(); return
        h.send_response(200); h.end_headers()
```

Replace `_publish_done` to set the queryable event instead of just printing:

```python
    def _publish_done(self):
        reason = (self.session.fail_reason
                  if self.session.state.value == 'FAILED' else None)
        ev = {'event': 'nav_done',
              'state': self.session.state.value,
              'goal_label': self.session.goal_label,
              'goal_id': self.session.goal_id,
              'reason': reason}
        print(f'NAV result: {ev}')
        with self._nav_done_lock:
            self._nav_done_event = ev
        if self.session.state.value == 'FAILED':
            try:
                self.http.post(self.cmd_url,
                               json={'action': 'nav_stop'}, timeout=0.5)
            except Exception:
                pass
        self.session = None
```

- [ ] **Step 2: Bot — new constants + helpers**

In `laptop/telegram_bot.py`, add after the existing URL constants:

```python
NAV_API = 'http://127.0.0.1:8092'

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, Update)
from telegram.ext import CallbackQueryHandler
```

Add helper functions:

```python
def fetch_candidates(cls):
    try:
        r = _session.post(f'{NAV_API}/nav/candidates',
                          json={'target_class': cls}, timeout=3.0)
        return r.json().get('candidates', [])
    except requests.RequestException:
        return []


def start_goto_id(lid):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'landmark_id': lid}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_goto_tag(tid):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'tag_id': tid}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_goto_xy(x, y, label):
    try:
        r = _session.post(f'{NAV_API}/nav/goto',
                          json={'xy': [x, y], 'label': label}, timeout=3.0)
        return r.status_code == 200
    except requests.RequestException:
        return False


def poll_nav_done():
    try:
        r = _session.get(f'{NAV_API}/nav/done', timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            return data if data else None
    except requests.RequestException:
        pass
    return None
```

- [ ] **Step 3: Handle `goto` actions in `on_message`**

Replace the section after `if cmd['action'] == 'map':` in `on_message` to also handle goto + patrol + room_tour. Insert these branches before `if cmd['action'] == 'find' and cmd['target'] is None:`:

```python
    if cmd['action'] == 'goto':
        if cmd.get('origin'):
            ok = start_goto_tag(0)
            if not ok:
                await context.bot.send_message(chat_id, '原点 (tag 0) 未在地图')
            else:
                await context.bot.send_message(chat_id, '前往原点 ...')
                threading.Thread(target=_poll_nav, args=(
                    __import__('asyncio').get_running_loop(),
                    context, chat_id), daemon=True).start()
            return
        if 'landmark_id' in cmd:
            ok = start_goto_id(cmd['landmark_id'])
            if not ok:
                await context.bot.send_message(
                    chat_id, f"id {cmd['landmark_id']} 未在地图")
            else:
                await context.bot.send_message(
                    chat_id, f"前往 landmark id {cmd['landmark_id']} ...")
                threading.Thread(target=_poll_nav, args=(
                    __import__('asyncio').get_running_loop(),
                    context, chat_id), daemon=True).start()
            return
        if 'target_class' in cmd:
            cands = fetch_candidates(cmd['target_class'])
            if not cands:
                await context.bot.send_message(
                    chat_id, f"未在地图中找到 {cmd['target_class']}")
                return
            if len(cands) == 1:
                lid = cands[0]['id']
                start_goto_id(lid)
                await context.bot.send_message(
                    chat_id, f"前往 {cmd['target_class']} id {lid} ...")
                threading.Thread(target=_poll_nav, args=(
                    __import__('asyncio').get_running_loop(),
                    context, chat_id), daemon=True).start()
                return
            # multi-candidate inline keyboard
            buttons = [[InlineKeyboardButton(
                f"{c['label']} id {c['id']} ({c['dist_m']:.1f}m)",
                callback_data=f"goto:{c['id']}")] for c in cands]
            await context.bot.send_message(
                chat_id, f"找到 {len(cands)} 个 {cmd['target_class']}",
                reply_markup=InlineKeyboardMarkup(buttons))
            return

    if cmd['action'] == 'patrol':
        await context.bot.send_message(chat_id, '开始巡逻 ...')
        threading.Thread(target=_run_patrol, args=(
            __import__('asyncio').get_running_loop(), context, chat_id),
            daemon=True).start()
        return

    if cmd['action'] == 'room_tour':
        await context.bot.send_message(chat_id, '开始绕室 ...')
        threading.Thread(target=_run_room_tour, args=(
            __import__('asyncio').get_running_loop(), context, chat_id),
            daemon=True).start()
        return
```

- [ ] **Step 4: Callback query handler for inline buttons**

Add (just below `on_message`):

```python
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.from_user.id not in AUTHORIZED_IDS:
        return
    data = q.data or ''
    if data.startswith('goto:'):
        lid = int(data.split(':', 1)[1])
        if start_goto_id(lid):
            await q.edit_message_text(f'前往 landmark id {lid} ...')
            threading.Thread(target=_poll_nav, args=(
                __import__('asyncio').get_running_loop(), context,
                q.message.chat.id), daemon=True).start()
        else:
            await q.edit_message_text('启动导航失败')
```

Register it in `main()`:

```python
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CallbackQueryHandler(on_callback))
```

- [ ] **Step 5: `_poll_nav` worker**

Add:

```python
def _poll_nav(loop, context, chat_id, stop_after_sec=300):
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline:
        time.sleep(0.7)
        ev = poll_nav_done()
        if ev is None:
            continue
        state = ev.get('state')
        label = ev.get('goal_label') or '?'
        gid = ev.get('goal_id')
        reason = ev.get('reason')
        if state == 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f'已到达 {label}' + (f' (id {gid})' if gid is not None else '')),
                loop)
        else:
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f'导航失败 ({reason or state})'), loop)
        return
```

- [ ] **Step 6: Compile-check + commit (patrol / chain follow in tasks 8-9)**

```bash
python -m py_compile laptop/telegram_bot.py
git add laptop/laptop_detector.py laptop/telegram_bot.py
git commit -m "feat(bot): goto-by-id/class with inline-keyboard disambiguation"
```

Stub `_run_patrol` / `_run_room_tour` as `pass` for now to keep compile clean; Task 9 implements them.

```python
def _run_patrol(loop, context, chat_id):
    pass


def _run_room_tour(loop, context, chat_id):
    pass
```

---

## Task 8: SEARCH → NAV chain

**Files:**
- Modify: `laptop/telegram_bot.py`

- [ ] **Step 1: Add a chain timer to the existing `_poll_events`**

The existing `_poll_events` already listens for `arrived:<class>`. Add a parallel branch: when launched as a `chain` (separate worker), if the search times out without `arrived`, query candidates and fall back to NAV.

Add a new worker:

```python
SEARCH_TIMEOUT_SEC = 30.0


def _run_chain(loop, context, chat_id, target):
    """SEARCH first; on timeout, fall back to NAV via the nav API."""
    import asyncio
    deadline = time.time() + SEARCH_TIMEOUT_SEC
    arrived = False
    while time.time() < deadline:
        time.sleep(0.7)
        st = get_status()
        if st is None:
            continue
        ev = st.get('event', '')
        if ev.startswith('arrived:'):
            arrived = True
            target_name = ev.split(':', 1)[1]
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id, f'已到达 {target_name}'), loop)
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id, f'到达 {target_name}'), loop)
            return
    if arrived:
        return
    # SEARCH timed out — try NAV fallback
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id,
            f'{SEARCH_TIMEOUT_SEC:.0f}s 未发现 {target}, 改用地图导航'), loop)
    cands = fetch_candidates(target)
    if not cands:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id,
                f'地图中也没有 {target} — 放弃'), loop)
        return
    # pick nearest CONFIRMED
    start_goto_id(cands[0]['id'])
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id,
            f"前往 {target} id {cands[0]['id']} ..."), loop)
    _poll_nav(loop, context, chat_id)
```

- [ ] **Step 2: Route `find` to the chain worker**

In `on_message`, find the existing `if cmd['action'] == 'find':` block (after `post_command(cmd)` succeeds). Replace its thread spawn from `_poll_events` to `_run_chain`:

```python
    if cmd['action'] == 'find':
        await context.bot.send_message(chat_id, f"前往寻找 {cmd['target']} ...")
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_run_chain, args=(loop, context, chat_id,
                                                  cmd['target']),
                         daemon=True).start()
```

The existing rotate_photo branch still uses `_poll_events` — leave it.

- [ ] **Step 3: Compile-check + commit**

```bash
python -m py_compile laptop/telegram_bot.py
git add laptop/telegram_bot.py
git commit -m "feat(bot): SEARCH->NAV chain fallback on search timeout"
```

---

## Task 9: Patrol + room-tour orchestration

**Files:**
- Modify: `laptop/telegram_bot.py`
- Modify: `laptop/laptop_detector.py`

- [ ] **Step 1: Bot queries the laptop for confirmed landmarks + tag list**

Add to `SlamRunner._start_nav_api`:

```python
                elif self.path == '/nav/landmarks':
                    runner._handle_landmarks_list(self)
                elif self.path == '/nav/tags':
                    runner._handle_tags_list(self)
```

(Inside the existing `do_GET` — extend the if/elif chain to add these endpoints alongside `/nav/done`.)

Add handler methods:

```python
    def _handle_landmarks_list(self, h):
        import json as _json
        snap = self.smap.snapshot(confirmed_only=True)
        body = _json.dumps({'landmarks': snap}).encode()
        h.send_response(200); h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(body))); h.end_headers()
        h.wfile.write(body)

    def _handle_tags_list(self, h):
        import json as _json
        tags = [{'id': i, 'x': float(e['x']), 'y': float(e['y'])}
                for i, e in self.tag_map.items()]
        body = _json.dumps({'tags': tags}).encode()
        h.send_response(200); h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(body))); h.end_headers()
        h.wfile.write(body)
```

- [ ] **Step 2: Replace bot patrol/room-tour stubs with real workers**

In `telegram_bot.py`, replace the stub `_run_patrol` and `_run_room_tour`:

```python
def _fetch_list(path):
    try:
        r = _session.get(f'{NAV_API}/{path}', timeout=3.0)
        return r.json()
    except requests.RequestException:
        return {}


def _wait_for_nav_done(timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        time.sleep(0.7)
        ev = poll_nav_done()
        if ev:
            return ev
    return None


def _run_patrol(loop, context, chat_id):
    import asyncio
    data = _fetch_list('nav/landmarks')
    landmarks = data.get('landmarks', [])
    if not landmarks:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, '地图中没有 CONFIRMED 物体'), loop)
        return
    for lm in landmarks:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id,
                f"前往 {lm['label']} id {lm['id']} ..."), loop)
        if not start_goto_id(lm['id']):
            continue
        ev = _wait_for_nav_done(180.0)
        if ev is None or ev.get('state') != 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f"跳过 {lm['label']} id {lm['id']} ({ev.get('reason') if ev else 'timeout'})"),
                loop)
            continue
        # photo at each stop
        asyncio.run_coroutine_threadsafe(
            _send_photo(context, chat_id,
                        f"{lm['label']} id {lm['id']}"), loop)
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id, '巡逻完成'), loop)


def _run_room_tour(loop, context, chat_id):
    import asyncio
    data = _fetch_list('nav/tags')
    tags = sorted(data.get('tags', []), key=lambda t: t['id'])
    if not tags:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, 'tag_map 为空'), loop)
        return
    for t in tags:
        asyncio.run_coroutine_threadsafe(
            context.bot.send_message(chat_id, f"前往 tag {t['id']} ..."), loop)
        if not start_goto_tag(t['id']):
            continue
        ev = _wait_for_nav_done(180.0)
        if ev is None or ev.get('state') != 'ARRIVED':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id,
                    f"跳过 tag {t['id']} ({ev.get('reason') if ev else 'timeout'})"),
                loop)
            continue
    asyncio.run_coroutine_threadsafe(
        context.bot.send_message(chat_id, '绕室完成'), loop)
```

- [ ] **Step 3: Make `SemanticMap.snapshot(confirmed_only=True)` available**

Check whether `slam/semantic_map.py` already supports `confirmed_only`. (It does — `def snapshot(self, confirmed_only: bool = False)`. If your tree differs, add the parameter.)

- [ ] **Step 4: Compile-check + commit**

```bash
python -m py_compile laptop/telegram_bot.py laptop/laptop_detector.py
git add laptop/telegram_bot.py laptop/laptop_detector.py
git commit -m "feat(bot): patrol + room_tour orchestration via nav API"
```

---

## Task 10: README + spec cross-link

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the Run section in `README.md`**

Insert under the existing `### Add semantic SLAM` section, before `### View the map`:

```markdown
### Add navigation (semantic-nav P2.1)

With SLAM enabled, navigation commands work over Telegram:

- `去 id 3` — drive to a specific landmark id.
- `回原点` — drive to tag 0.
- `去椅子` — drive to the nearest confirmed `chair`; if multiple, the bot
  asks via inline buttons.
- `去找椅子` — try visual SEARCH first; on timeout fall back to NAV using
  any confirmed `chair` on the map.
- `巡逻` — visit every confirmed landmark, photo at each.
- `绕室` — visit every AprilTag by id.
- `停` — aborts the active goto.

Tunable from `laptop_detector.py`: `--arrived-radius-m`, `--nav-v-max`,
`--nav-w-max`, `--max-pulse-sec`, `--no-tag-grace-sec`,
`--dr-distance-limit-m`, `--dr-time-limit-sec`, `--block-retries`,
`--scan-max-rotations`, `--landmark-conf-min`. Defaults match
[`docs/superpowers/specs/2026-05-17-semantic-nav-design.md`](docs/superpowers/specs/2026-05-17-semantic-nav-design.md).
```

- [ ] **Step 2: Topic / endpoint table**

In the "HTTP endpoints" table, append a row:

```markdown
| `http://<laptop>:8092/nav/*` | `laptop_detector.py --slam` | nav api for telegram_bot (candidates, goto, done) |
```

- [ ] **Step 3: Remove obsolete dead-reckoning + nav TODO bullets**

Delete the two TODO bullets covered by P2.1:
- "Dead-reckoning between tag observations."
- (Keep the side-IR bullet — phase 2 uses it but the bullet text overlaps; rewrite to:)

```markdown
- **Side-IR is now consumed by NAV mode for blocked events** but is not
  yet used in the SEARCH/APPROACHING discrete-pulse loop. Wiring it into
  SEARCH would let the legacy reactive search also avoid obstacles.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document semantic-nav P2.1 commands and endpoints"
```

---

## Final verification

- [ ] **Step 1: Full laptop test suite**

```bash
cd laptop
python -m pytest tests/ -q
```

Expected: all tests pass (existing 35 + new tests from tasks 1-3 and 6).

- [ ] **Step 2: All compile**

```bash
cd ..
python -m py_compile laptop/slam/landmark_selector.py \
    laptop/slam/dead_reckoner.py \
    laptop/slam/nav_session.py \
    laptop/laptop_detector.py \
    laptop/telegram_bot.py \
    laptop/nl_parser.py \
    src/embodied_mvp/embodied_mvp/agent_node.py
```

Expected: silent success.

- [ ] **Step 3: Hardware checklist (manual)**

Run through the spec's hardware test list:
1. `colcon build` on Pi, restart launch with `enable_semantic_map:=true`.
2. `python laptop_detector.py --pi 172.20.10.4 --slam` (now also serves nav api).
3. `python telegram_bot.py`.
4. Telegram: `回原点`, `去 id 0`, `去椅子`, `去找椅子`, `巡逻`, `绕室`.
5. Block a path manually mid-drive → confirm side-step.
6. Drive through a tag-blind region → confirm dead-reckoning + scan recovery.
7. `停` mid-drive → robot halts within ≤ 1 pulse.
