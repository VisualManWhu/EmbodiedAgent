# Semantic Navigation — Phase 2.1 Design

## Context

EmbodiedAgent now has a working semantic map: AprilTag anchors define the
`map` frame, the robot localizes off any visible tag, and multi-view-fused
COCO object landmarks are stored with TENTATIVE/CONFIRMED state. See
[`docs/CONTEXT.md`](../../CONTEXT.md) and the phase-1 design.

The next phase is **navigation that uses the semantic map**: given an object
class or landmark id, drive the robot to that landmark in the map frame,
handling the gaps in localization and obstacles along the way.

The existing SEARCH FSM is purely reactive (visual servo on whatever YOLO
sees right now). It cannot reach objects that aren't currently visible. This
spec adds a map-aware NAV mode that closes that gap and a chain behavior
that combines SEARCH and NAV.

## Hard constraints (inherited)

- No wheel encoders, no IMU. The only positional source is AprilTag PnP,
  and only when a tag is in view.
- Mecanum motion is open-loop and slip-prone — commanded distance does not
  equal travelled distance.
- Pi5 + 4-motor battery shares one BMS. Heavy compute stays off the Pi.
- The semantic map lives on the laptop; the Pi receives it for RViz only.

## Decisions locked during brainstorming

| Decision | Choice |
|---|---|
| Phase scope | Full P2.1: by-id, by-class (with disambiguation), patrol, room-tour, chain (SEARCH→NAV fallback) |
| Planner location | Hybrid — high-level loop on laptop, low-level reactive avoidance on Pi |
| No-tag handling | Short cmd_vel dead-reckoning (cap 5 s / 0.5 m) plus active pan-tilt scan |
| Arrival criterion | `dist(robot_xy, goal_xy) < arrived_radius` (default 0.4 m) |
| Class disambiguation | Telegram inline-keyboard prompt listing candidates |
| Chain behavior | `去找<class>` runs SEARCH first; if it times out (default 30 s) and a CONFIRMED landmark of that class exists, fall back to NAV |

## Architecture

The laptop runs the navigation state machine alongside the existing detector
and SLAM mapper. The Pi gets a thin new `agent_node` mode that executes
single motion pulses and reports completion or obstacles back. The map and
all planning logic stay on the laptop; the Pi's CPU stays light.

```
Phone (Telegram)
  | Telegram
Laptop:
  telegram_bot.py
    - inline-keyboard disambiguation for multi-candidate class queries
    - chain controller: SEARCH first, fallback to NAV on timeout
    - holds per-chat state (awaiting_disambiguation, chain phase)
  laptop_detector.py (existing main loop)
    - per-frame: SlamRunner.process (existing)
    - per-frame: NavSession.tick if a goto is active
  slam/nav_session.py
    - single-goto state machine; produces motion pulses from pose + goal
  slam/landmark_selector.py
    - class -> candidate list (CONFIRMED, conf >= threshold, optional radius)
    - sorts by distance from current robot pose
  slam/dead_reckoner.py
    - integrates issued (vx, wz, seconds) pulses to extrapolate pose when
      no tag is visible; resets on each tag fix
  | HTTP
Pi5 (ROS2):
  agent_node, new NAV mode
    - accepts {action: 'nav_pulse', vx, vy, wz, seconds}
    - while pulse runs: if /ir/range < stop_d or /ir/left or /ir/right hit
        -> abort pulse, post event "blocked:front|left|right"
    - on natural pulse completion, post event "pulse_done"
    - existing /command and /status HTTP endpoints carry the traffic
```

### Why this split

The laptop already pulls every camera frame, runs YOLO and AprilTag
detection, and holds the semantic map. Doing planning there reuses the same
pose pipeline NavSession needs and avoids syncing the map to the Pi. Keeping
real-time obstacle reaction on the Pi means a `blocked` decision happens
within the Pi control loop (≈10 Hz) rather than waiting for a laptop
round-trip — important for collision avoidance even with ~1 s of laptop
latency budget.

## Components

### `laptop/slam/nav_session.py`

A single class `NavSession` driving one navigation goal end to end.

- Construction: `NavSession(goal_xy, goal_id, goal_label, config)`.
- `tick(robot_pose, now)` returns `NavCommand | None`:
  - `None` if waiting on a Pi event or arrived.
  - `NavCommand` with `(vx, vy, wz, seconds, kind)` to POST to the Pi.
- `on_pi_event(event)` consumes `pulse_done` / `blocked:<side>` / `arrived`.
- States: `ALIGNING`, `DRIVING`, `WAITING_PULSE`, `AVOIDING`, `SCANNING`,
  `ARRIVED`, `FAILED`.
- Heading-then-drive pulse loop:
  - `heading_err = wrap(atan2(dy, dx) - yaw)`.
  - `|heading_err| > heading_tol (0.35 rad)` → rotate pulse,
    `seconds = clamp(|heading_err| / w_max, 0.2, 1.0)`.
  - else → forward pulse,
    `seconds = clamp(dist / v_max, 0.2, 1.5)`.
- Avoidance: on `blocked:<side>`, queue a mecanum side-step sequence
  (strafe away 0.6 s using `vy = ±strafe_speed` → forward 0.8 s → resume).
  This is the first autonomous use of mecanum strafe; `motor_node` already
  supports it. 3 consecutive failures → `FAILED("blocked")`.
- Scan: when no tag has updated the pose for `no_tag_grace_sec` (5 s) AND
  the dead-reckoner has accumulated `dr_distance_limit_m` (0.5 m), pause
  driving and emit a pan-tilt sweep command (`±60°` yaw over 4 s). If a tag
  is still missing after the sweep, `FAILED("lost")`.

### `laptop/slam/landmark_selector.py`

Pure functions over `SemanticMap`:

- `by_id(smap, id) -> Landmark | None`.
- `by_class(smap, cls, conf_min=0.7, state='CONFIRMED') -> list[Landmark]`,
  ordered by ascending distance from a given robot xy.
- `by_tag(tag_map, tag_id) -> (x, y)`.
- `nearest_by_class(...)` — convenience wrapping the above; returns the
  single best candidate or None.

### `laptop/slam/dead_reckoner.py`

Integrates issued motion pulses to bridge tag-fix gaps.

- `record_pulse(vx, vy, wz, seconds, started_at)` — appends to a buffer.
- `pose_at(now, last_tag_pose)` — replays pulses from `last_tag_pose`,
  integrating mecanum-frame velocities into the world frame, returning an
  estimated pose. Hard caps:
  - if `(now - last_tag_pose_time) > dr_time_limit_sec` (5 s) → return
    `last_tag_pose, stale=True`.
  - if accumulated `||dx, dy||` > `dr_distance_limit_m` (0.5 m) → likewise.
- `reset(new_tag_pose, t)` on every fresh tag fix.

Open-loop accuracy is poor but bounded by the small window; this exists to
let the robot traverse a short region without an in-view tag (e.g. between
two wall-mounted tags) rather than freeze entirely.

### `laptop/telegram_bot.py` extensions

- Per-chat state dict keyed by `chat_id`:
  - `awaiting_disambiguation`: `{chat_id: {'candidates': [...], 'verb': ...}}`.
  - `chain_in_flight`: `{chat_id: {'class': ..., 'started_at': ..., 'phase': 'search'|'nav'}}`.
- On `去<class>` with multiple candidates, send an inline keyboard listing
  `chair (id 3, 0.8 m)`, `chair (id 7, 2.1 m)`, ... ; callback data carries
  the chosen id; the bot then issues the goto.
- Chain (`去找<class>`):
  - Phase `search`: post the existing `find` command, start a 30 s timer.
  - On `arrived:<class>` event → done.
  - On timeout: query laptop for known landmarks of that class. If any
    CONFIRMED present → start NAV; else → reply "未找到 X, 地图也没有".

### `laptop/nl_parser.py` extensions

- `去 id <N>` → `{'action': 'goto', 'landmark_id': N}`.
- `回原点` / `回到原点` → `{'action': 'goto', 'origin': True}`.
- `去 <class>` / `goto <class>` → `{'action': 'goto', 'target_class': class}`.
- `去找 <class>` / `find <class>` (existing) — bot handles chain.
- `巡逻` / `patrol` → `{'action': 'patrol'}` — visits each CONFIRMED
  landmark, photo at each. Order is greedy-nearest-from-current-pose
  recomputed at each leg (no global TSP — landmarks may move between phases
  and the operator can interrupt with `停` mid-tour).
- `绕室` / `room tour` → `{'action': 'room_tour'}` — visits each tag by id.
- `停` aborts an active NAV (existing `stop` action).

### Pi `agent_node` NAV mode

- New action accepted by `command_server`:
  ```
  {"action": "nav_pulse", "vx": 0.0, "vy": 0.0, "wz": 0.4, "seconds": 0.5}
  ```
  Setting `mode = 'NAV'`, executing one pulse with side-IR / ultrasonic
  monitoring at the regular 10 Hz tick. On natural completion of `seconds`,
  post event `pulse_done`. On obstacle, abort and post
  `blocked:front` (ultrasonic), `blocked:left`, or `blocked:right`
  (side IR). After abort the robot holds still in `NAV` until the next
  command.
- New action `nav_stop` returns to `IDLE`.
- `nav_pulse` reuses the existing pulse-clamp / slew-limit logic. `vy` is
  non-zero only during avoidance side-steps; forward driving keeps `vy = 0`
  for predictability.

## Data flow per tick (laptop main loop)

1. Pull frame + pan/tilt headers from Pi `/snapshot` (existing).
2. Run YOLO + AprilTag (existing).
3. If an AprilTag pose is recovered: update SlamRunner, `DeadReckoner.reset`.
4. If a `NavSession` is active:
   1. Build `robot_pose`: tag pose if fresh, else `DeadReckoner.pose_at`.
   2. `cmd = nav_session.tick(robot_pose, now)`.
   3. If `cmd` is not None: POST to Pi `/command` and
      `DeadReckoner.record_pulse(cmd, now)`.
5. Poll Pi `/status`; route `pulse_done` / `blocked:*` to
   `nav_session.on_pi_event`.
6. If `nav_session.state == ARRIVED`: post Telegram completion message,
   clear the session. If `FAILED`: report the reason and clear.

## Telegram interactions

### Simple goto by id
```
User : 去 id 3
Bot  : 前往 chair (id 3) at (1.2, 0.6) ...
... robot drives ...
Bot  : 已到达 chair (id 3)
```

### Class with disambiguation
```
User : 去椅子
Bot  : 找到 3 个 chair:                [inline buttons]
       [chair id 3 (0.8 m)] [chair id 7 (2.1 m)] [chair id 12 (3.5 m)]
User : (taps chair id 3)
Bot  : 前往 chair (id 3) ...
Bot  : 已到达 chair (id 3)
```

### Chain SEARCH → NAV
```
User : 去找椅子
Bot  : 开始搜索 chair ...
... SEARCH runs, no chair visible ...
Bot  : 30s 未发现 chair, 改用地图导航
... NAV runs ...
Bot  : 已到达 chair (id 3)
```

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| No matching landmark | selector returns empty | "未在地图中, 请先建图" |
| Multiple candidates | `len(candidates) > 1` | Telegram inline keyboard |
| Path blocked 3 × in a row | side-step counter | `FAILED("blocked")`, "路径阻塞, 人工接管" |
| Tag lost > 5 s and scan failed | scan returns no detection | `FAILED("lost")`, "丢失定位" |
| Dead-reckon exceeds cap | distance/time limit hit | Force scan immediately |
| Pi unreachable | POST timeout | Bot replies "小车失联" (existing) |
| User sends `停` mid-nav | parser → `stop` | `nav_pulse` cancelled, mode → IDLE |

## Testing

### Unit (laptop, no hardware)

- `landmark_selector`:
  - Empty map → `nearest_by_class` returns None.
  - Multiple chairs with mixed `state` and `conf` → returns nearest
    `CONFIRMED` with `conf ≥ threshold`; falls through TENTATIVE.
  - Tie on distance broken deterministically by id.
- `dead_reckoner`:
  - Forward 1 m/s × 1 s starting at (0, 0, 0) → pose ≈ (1, 0, 0).
  - Pure rotation: 1 rad/s × 1 s yaw → 1 rad.
  - Composed forward + rotation matches numerical integration.
  - Time cap > 5 s → returns last_tag_pose with `stale=True`.
- `NavSession.tick`:
  - Pose far + bad heading → rotate pulse.
  - Pose far + heading OK → forward pulse.
  - Pose within radius → `ARRIVED`.
  - `blocked:front` event → queues side-step sequence.
  - 3 consecutive `blocked` → `FAILED("blocked")`.
- `nl_parser`:
  - `去 id 3` → `{action: goto, landmark_id: 3}`.
  - `回原点` → `{action: goto, origin: True}`.
  - `巡逻` → `{action: patrol}`.

### Integration (laptop, synthetic)

- Drive a synthetic SemanticMap + scripted tag-pose stream through
  NavSession + DeadReckoner; assert the robot reaches the goal within the
  arrived radius in N pulses.
- Simulate a `blocked:left` mid-drive → asserts side-step subsequence.
- Tag drops out mid-drive → DeadReckoner extrapolates; on re-acquire,
  `NavSession` re-aligns to actual pose.

### Hardware (manual checklist)

- `去 id 0` from various start poses → arrives within 0.4 m.
- `去椅子` with 2 chairs mapped → inline buttons appear; selection works.
- `去找椅子` chain: with chair removed → SEARCH times out, NAV resumes.
- Manually block path → robot side-steps and continues.
- Drive through a known tag gap → dead-reckon bridges; scan recovers.
- `停` during NAV → robot stops within a pulse.

## Parameters and defaults

Exposed as CLI flags on `laptop_detector.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--arrived-radius-m` | 0.4 | Distance for ARRIVED |
| `--heading-tol-rad` | 0.35 | Pure-rotation threshold |
| `--nav-v-max` | 0.15 | Forward speed in NAV |
| `--nav-w-max` | 0.4 | Yaw speed in NAV |
| `--max-pulse-sec` | 1.5 | Pulse seconds cap |
| `--no-tag-grace-sec` | 5.0 | Tolerate missing tag before scan |
| `--dr-distance-limit-m` | 0.5 | Dead-reckon hard distance cap |
| `--dr-time-limit-sec` | 5.0 | Dead-reckon hard time cap |
| `--scan-sweep-deg` | 60 | ± pan sweep for tag scan |
| `--search-timeout-sec` | 30 | Chain SEARCH timeout before NAV fallback |
| `--landmark-conf-min` | 0.7 | Selector confidence floor |
| `--block-retries` | 3 | Side-step retries before FAILED |

## Out of scope (later phases)

- Visual-arrival confirmation (using YOLO bbox after NAV stops near goal).
- Auto-built tag pose graph and tag-graph multi-hop planning (Option B in
  the brainstorm).
- Occupancy grid / Nav2.
- Multi-room semantic queries ("the chair in the kitchen").
- Drawing planned paths onto the top-down map PNG.
- Active obstacle re-routing beyond fixed-shape side-step.

## Implementation order

1. `landmark_selector` + tests.
2. `dead_reckoner` + tests.
3. `NavSession` core (state machine, pulse generation) + tests.
4. Pi `agent_node` NAV mode + `nav_pulse` action + event posting.
5. Integration into `laptop_detector.py` main loop.
6. `nl_parser` additions + tests.
7. Telegram inline-keyboard disambiguation flow.
8. Chain controller (SEARCH timeout → NAV fallback).
9. Patrol / room-tour multi-goto orchestrator.
10. CLI flags wiring; defaults documented in README.
11. Hardware checklist run-through.
