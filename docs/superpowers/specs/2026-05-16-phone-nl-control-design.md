# Design — Phone Natural-Language Control

## Context

The Pi5 mecanum car already does autonomous object search (offloaded detection
on the laptop GPU). This feature adds **remote control from a phone via natural
language**, so the operator can drive the car, request photos, and direct it to
find/approach arbitrary COCO objects on the fly.

Workflow it enables: car sends a photo → operator spots a bottle → tells the car
to find the bottle → car approaches and arrives → auto-sends a photo → operator
spots a chair → tells the car to find the chair → and so on.

## Requirements (settled)

1. Comms channel: **Telegram bot** (phone uses the Telegram app).
2. NL parsing: **keyword rule matching** (offline, no LLM).
3. Bot process runs on the **laptop** (alongside `laptop_detector.py`).
4. Motion commands move a **fixed small step** per command.
5. A new command **immediately interrupts** the current task.
6. On arrival at a target: **notify + auto-send one photo**.
7. "旋转拍照" command: rotate in place, take 4 photos (~90° apart), end back
   at roughly the original heading. Return accuracy is **approximate**
   (open-loop, no encoder/IMU) — acceptable.

## Architecture

```
Phone (Telegram app)
   |  Telegram
Laptop: telegram_bot.py     — bot + keyword parser
        laptop_detector.py  — unchanged (feeds /detections for SEARCH)
   |  HTTP (LAN)
Pi5:    agent_node          — IDLE/MANUAL/SEARCH, sole /cmd_vel owner, HTTP server
        camera / motor / ir / side_ir / pantilt / det_bridge / mjpeg  — unchanged
```

- `telegram_bot.py` is the only component that talks to Telegram. It parses
  messages, POSTs commands to the Pi, pulls snapshots, and polls status.
- `agent_node` is the only component that publishes `/cmd_vel` — no contention,
  and mode switching is the interrupt mechanism.
- Photos do not involve `agent_node`: the bot GETs `mjpeg_node`'s `/snapshot`
  directly.

## Component 1 — `agent_node` (Pi)

Evolves the current `search_node` into a mode-based behavior supervisor.

**Modes:**

| Mode   | Behavior                                              | /cmd_vel        |
|--------|-------------------------------------------------------|-----------------|
| IDLE   | stand by                                              | publishes 0     |
| MANUAL | execute one timed fixed-step move, then revert to IDLE| fixed twist until deadline |
| SEARCH | existing search FSM (SEARCHING/APPROACHING/ARRIVED as internal sub-states) | existing logic |
| ROTATE_PHOTO | 4× { still dwell + emit `photo_ready:N` } separated by ~90° timed rotations, then IDLE | dwell=0, rotate=fixed wz |

The SEARCH sub-state machine and all approach/scan/arrival logic is the current
`search_node` behavior, unchanged — it just becomes the SEARCH mode body.

**ROTATE_PHOTO sequence:** for N = 1..4 — hold still for `photo_dwell_sec`
(sharp frame) while exposing `event = photo_ready:N`, then rotate in place for
`rotate_90_sec` (timed, open-loop ≈ 90°). After the 4th photo and 4th rotation
the car has turned ≈ 360°, back to roughly the original heading; mode → IDLE,
`event = rotate_photo_done`. Return heading is approximate (no odometry).

**HTTP server** (port 9091, `ThreadingHTTPServer`, same pattern as
`det_bridge_node` / `mjpeg_node`):

- `POST /command` — body `{"action": ...}`. Applying a command **immediately
  switches mode**, which terminates whatever was running (the interrupt).
  - `{"action":"move","dir":"forward|backward|left|right|rotate_left|rotate_right"}`
    → MANUAL: set a fixed twist (forward/backward→vx, left/right→vy,
    rotate_*→wz) with `deadline = now + manual_step_sec`.
  - `{"action":"stop"}` → IDLE.
  - `{"action":"find","target":"<coco_class>"}` → SEARCH with `target_class`
    set at runtime.
  - `{"action":"rotate_photo"}` → ROTATE_PHOTO.
- `GET /status` — returns `{"mode": "...", "event": "..."}`. `event` carries
  one-shot notices, e.g. `arrived:bottle`, `photo_ready:2`, `rotate_photo_done`,
  consumed/cleared after it is read.

**Watchdog:** in MANUAL, when `now > deadline` → revert to IDLE and publish 0.
`motor_node`'s existing 0.5 s `/cmd_vel` timeout is the lower-level failsafe.

**Params (params.yaml `agent_node`):** `command_port: 9091`,
`manual_step_sec: 1.0`, `rotate_90_sec` (calibrated empirically — duration of a
~90° in-place turn), `photo_dwell_sec: 2.0`, plus all existing search params.
`target_class` becomes a runtime-settable default rather than a fixed launch arg.

**Single /cmd_vel owner:** modes are mutually exclusive; a new command overwrites
`mode`, so the previous task ends cleanly with no arbitration needed.

## Component 2 — `telegram_bot.py` (laptop)

New pure-Python file under `laptop/`. Dependency: `python-telegram-bot`.

**Responsibilities:**

1. Receive Telegram messages → keyword-parse → POST to Pi `:9091/command`.
2. "photo" → GET Pi `:8080/snapshot` → send the image back via Telegram.
3. While a `find` is active → background poll Pi `:9091/status` → on
   `event == arrived:<X>` → send "已到达 X" + auto-fetch a snapshot and send it.
4. While a `rotate_photo` is active → poll Pi `:9091/status` → on each
   `event == photo_ready:N` → GET snapshot, send to phone labelled photo N of 4
   (前/右/后/左); on `rotate_photo_done` → send "旋转拍照完成".

**Keyword parse table (Chinese → command):**

| Phone text                       | Parsed command            |
|-----------------------------------|---------------------------|
| 前 / 前进 / 往前                  | move forward              |
| 后 / 后退                         | move backward             |
| 左移 / 右移                       | move left / right (strafe)|
| 左转 / 右转                       | move rotate_left / right  |
| 停 / 停下 / 停止                  | stop                      |
| 拍照 / 照片 / 看看                | photo                     |
| 旋转拍照 / 环拍 / 转一圈拍照      | rotate_photo              |
| 找X / 去找X / 寻找X / 靠近X       | find, target=X            |

Parse order matters: `旋转拍照` contains `拍照`, so the `rotate_photo` keyword
must be tested before the plain `photo` keyword.

**Target name map (Chinese → COCO class):** small table for common objects —
`瓶子/水瓶→bottle`, `椅子→chair`, `人→person`, `杯子→cup`, `手机→cell phone`,
`笔记本/电脑→laptop`, `书→book`, … English COCO names also accepted directly.
Unknown target → bot replies "不认识该目标" + supported list.

**Config (top of script or a config block):** Telegram bot token (from
@BotFather), Pi IP, authorized Telegram user ID(s) — only the owner's account is
obeyed; messages from other users are ignored.

**Parse failure** → bot replies with a hint and the supported-command list.

## Data Flow — "find bottle" example

1. Phone: "去找瓶子" → bot parses → `{action:find, target:bottle}` → POST Pi.
2. `agent_node` → SEARCH mode, `target_class=bottle`.
3. Bot starts polling `:9091/status`.
4. `agent_node` runs the existing search/approach behavior (uses `/detections`
   from `laptop_detector.py`).
5. On arrival → `agent_node` sets `event=arrived:bottle`, mode → IDLE.
6. Bot sees the event → sends "已到达 bottle" + a snapshot to the phone.

## Error Handling / Edge Cases

- Bot↔Pi network failure → POST timeout → bot replies "小车失联".
- MANUAL deadline reached → auto-revert to IDLE (lost commands can't run away).
- `motor_node` 0.5 s `/cmd_vel` watchdog is the runaway failsafe.
- SEARCH target not found → existing behavior (keeps searching, non-terminal).
- Unparseable message → bot replies hint + command list.
- Unknown target name → bot replies "不认识" + supported list.
- Unauthorized Telegram user → bot ignores the message.
- Snapshot fetch fails → bot replies "取图失败".
- Any new command interrupts immediately; "停" takes effect at any time.

## Testing

1. **agent_node unit** — `curl POST /command` with move/stop/find; verify mode
   switches and `/cmd_vel` output.
2. **Interrupt** — POST `move` during an active `find`; verify immediate switch
   to MANUAL.
3. **Bot parsing** — feed varied Chinese sentences; verify parsed command
   (a `--dry-run` flag prints the parse without POSTing).
4. **End-to-end** — phone "前进" → car steps; "拍照" → photo received;
   "去找瓶子" → car searches, arrives, auto-sends photo; mid-task "停" → stops.
   "旋转拍照" → car turns in place, 4 labelled photos arrive, ends ≈ original
   heading, "旋转拍照完成" sent.
5. **Regression** — the original autonomous `mvp.launch.py` flow still works
   (agent_node defaults to IDLE; existing search behavior intact under SEARCH).

## Deliverables

- Change: `search_node.py` → `agent_node.py` (mode state machine + HTTP server);
  `setup.py`, `launch/mvp.launch.py`, `config/params.yaml`.
- New: `laptop/telegram_bot.py`.
- README: add a phone-control section (bot token setup, run instructions).

## Out of Scope (YAGNI)

- LLM-based parsing (keyword matching chosen).
- Distance/duration arguments in motion commands (fixed step chosen).
- Command queueing (new command interrupts instead).
- Multi-user / access control beyond a fixed authorized-ID allowlist.
