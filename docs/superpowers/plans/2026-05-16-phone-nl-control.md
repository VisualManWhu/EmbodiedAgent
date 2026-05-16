# Phone Natural-Language Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Telegram-bot phone control to the search robot — basic motion, photo capture, on-the-fly target search, and a rotate-and-photo macro.

**Architecture:** A laptop-side Telegram bot keyword-parses messages and POSTs commands to the Pi. On the Pi, `search_node` becomes `agent_node` — a mode-based behavior supervisor (IDLE / MANUAL / SEARCH / ROTATE_PHOTO) that is the sole `/cmd_vel` publisher; mode switching is the interrupt mechanism. A small HTTP server (`command_server.py`) carries commands in and status out.

**Tech Stack:** Python 3.11, ROS2 Humble (rclpy), `python-telegram-bot` (laptop), stdlib `http.server`, `pytest` (laptop parser tests).

**Spec:** `docs/superpowers/specs/2026-05-16-phone-nl-control-design.md`

---

## File Structure

**Pi — ROS2 package `src/embodied_mvp/`:**
- `embodied_mvp/command_server.py` *(new)* — thread-safe HTTP command/status server. Transport only, no behavior.
- `embodied_mvp/agent_node.py` *(renamed from `search_node.py`, then extended)* — mode FSM + the existing search behavior as one mode.
- `setup.py`, `launch/mvp.launch.py`, `config/params.yaml` *(modified)* — rename wiring + new params.

**Laptop — `laptop/`:**
- `nl_parser.py` *(new)* — pure keyword parser, `parse(text) -> command dict | None`.
- `telegram_bot.py` *(new)* — Telegram I/O, orchestration, status polling.
- `tests/test_nl_parser.py` *(new)* — pytest unit tests for the parser.

**Docs:** `README.md` *(modified)* — phone-control section.

---

## Task 1: NL keyword parser (laptop, TDD)

**Files:**
- Create: `laptop/nl_parser.py`
- Test: `laptop/tests/test_nl_parser.py`

- [ ] **Step 1: Install pytest on the laptop**

Run (Windows PowerShell): `pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Write the failing test**

Create `laptop/tests/test_nl_parser.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nl_parser import parse


def test_move_forward():
    assert parse('前进') == {'action': 'move', 'dir': 'forward'}
    assert parse('往前') == {'action': 'move', 'dir': 'forward'}


def test_move_backward():
    assert parse('后退') == {'action': 'move', 'dir': 'backward'}


def test_rotate_vs_strafe():
    assert parse('左转') == {'action': 'move', 'dir': 'rotate_left'}
    assert parse('右转') == {'action': 'move', 'dir': 'rotate_right'}
    assert parse('左移') == {'action': 'move', 'dir': 'left'}
    assert parse('右移') == {'action': 'move', 'dir': 'right'}


def test_stop():
    assert parse('停') == {'action': 'stop'}
    assert parse('停下来') == {'action': 'stop'}


def test_photo():
    assert parse('拍照') == {'action': 'photo'}
    assert parse('给我看看') == {'action': 'photo'}


def test_rotate_photo_not_photo():
    # '旋转拍照' contains '拍照' — must resolve to rotate_photo, not photo
    assert parse('旋转拍照') == {'action': 'rotate_photo'}
    assert parse('环拍') == {'action': 'rotate_photo'}


def test_find_known_target():
    assert parse('去找瓶子') == {'action': 'find', 'target': 'bottle'}
    assert parse('找椅子') == {'action': 'find', 'target': 'chair'}
    assert parse('靠近那个人') == {'action': 'find', 'target': 'person'}


def test_find_english_target():
    assert parse('find bottle') == {'action': 'find', 'target': 'bottle'}


def test_find_unknown_target():
    assert parse('去找独角兽') == {'action': 'find', 'target': None}


def test_unparseable():
    assert parse('帮我把灯关了') is None
    assert parse('') is None
```

- [ ] **Step 3: Run the test, verify it fails**

Run: `cd laptop && python -m pytest tests/test_nl_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nl_parser'`

- [ ] **Step 4: Write `laptop/nl_parser.py`**

```python
"""Keyword parser: Chinese / English text -> structured command dict.

parse(text) returns one of:
  {'action': 'move', 'dir': 'forward'|'backward'|'left'|'right'|'rotate_left'|'rotate_right'}
  {'action': 'stop'}
  {'action': 'photo'}
  {'action': 'rotate_photo'}
  {'action': 'find', 'target': '<coco_class>'}   # target None if unrecognised
  None  -- unparseable
"""

# Chinese alias -> COCO class name
TARGET_MAP = {
    '瓶子': 'bottle', '水瓶': 'bottle', '瓶': 'bottle',
    '椅子': 'chair', '椅': 'chair',
    '人': 'person',
    '杯子': 'cup', '杯': 'cup',
    '手机': 'cell phone',
    '笔记本': 'laptop', '电脑': 'laptop',
    '书': 'book',
    '键盘': 'keyboard',
    '鼠标': 'mouse',
    '背包': 'backpack', '包': 'backpack',
    '沙发': 'couch',
    '电视': 'tv',
    '钟': 'clock', '时钟': 'clock',
}

# English COCO class names accepted directly
COCO_CLASSES = {
    'person', 'bottle', 'cup', 'bowl', 'chair', 'couch', 'bed', 'tv',
    'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'clock', 'vase',
    'backpack', 'umbrella', 'handbag', 'remote', 'scissors', 'dining table',
}

SUPPORTED_TARGETS_CN = '瓶子 / 椅子 / 人 / 杯子 / 手机 / 笔记本 / 书 / 背包 / 沙发 / 电视'


def _match_target(text):
    low = text.lower()
    # longest Chinese alias first (so '水瓶' beats '瓶')
    for cn in sorted(TARGET_MAP, key=len, reverse=True):
        if cn in text:
            return TARGET_MAP[cn]
    for c in sorted(COCO_CLASSES, key=len, reverse=True):
        if c in low:
            return c
    return None


def parse(text):
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    # rotate_photo MUST be checked before photo ('旋转拍照' contains '拍照')
    if any(k in t for k in ('旋转拍照', '环拍', '转一圈拍照', '转圈拍照')):
        return {'action': 'rotate_photo'}

    if any(k in t for k in ('停', 'stop')):
        return {'action': 'stop'}

    if any(k in t for k in ('拍照', '照片', '看看', 'photo')):
        return {'action': 'photo'}

    if any(k in t for k in ('找', '寻找', '靠近', 'find')):
        return {'action': 'find', 'target': _match_target(t)}

    # rotation before plain strafe ('左转' contains '左')
    if any(k in t for k in ('左转', 'turn left', 'rotate left')):
        return {'action': 'move', 'dir': 'rotate_left'}
    if any(k in t for k in ('右转', 'turn right', 'rotate right')):
        return {'action': 'move', 'dir': 'rotate_right'}
    if any(k in t for k in ('左移', '向左', '左')):
        return {'action': 'move', 'dir': 'left'}
    if any(k in t for k in ('右移', '向右', '右')):
        return {'action': 'move', 'dir': 'right'}
    if any(k in t for k in ('前进', '往前', '向前', '前', 'forward')):
        return {'action': 'move', 'dir': 'forward'}
    if any(k in t for k in ('后退', '倒退', '后', 'backward', 'back')):
        return {'action': 'move', 'dir': 'backward'}

    return None
```

- [ ] **Step 5: Run the test, verify it passes**

Run: `cd laptop && python -m pytest tests/test_nl_parser.py -v`
Expected: PASS — all 10 tests green.

- [ ] **Step 6: Commit**

```bash
git add laptop/nl_parser.py laptop/tests/test_nl_parser.py
git commit -m "feat: add NL keyword parser for phone commands"
```

---

## Task 2: HTTP command/status server (Pi)

**Files:**
- Create: `src/embodied_mvp/embodied_mvp/command_server.py`
- Test: `src/embodied_mvp/tests/test_command_server.py`

- [ ] **Step 1: Write the failing test**

Create `src/embodied_mvp/tests/test_command_server.py`:

```python
import json
import time
import urllib.request

from embodied_mvp.command_server import CommandServer


def _post(port, body):
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/command',
        data=json.dumps(body).encode(), method='POST')
    return urllib.request.urlopen(req, timeout=2).read()


def _get_status(port):
    return json.loads(
        urllib.request.urlopen(f'http://127.0.0.1:{port}/status', timeout=2).read())


def test_command_roundtrip():
    srv = CommandServer(port=19099)
    try:
        time.sleep(0.2)
        _post(19099, {'action': 'stop'})
        assert srv.take_command() == {'action': 'stop'}
        # consumed -> next take is None
        assert srv.take_command() is None
    finally:
        srv.shutdown()


def test_status_and_event_oneshot():
    srv = CommandServer(port=19098)
    try:
        time.sleep(0.2)
        srv.set_mode('SEARCH')
        srv.post_event('arrived:bottle')
        s1 = _get_status(19098)
        assert s1['mode'] == 'SEARCH'
        assert s1['event'] == 'arrived:bottle'
        # event is one-shot — cleared after the first GET
        s2 = _get_status(19098)
        assert s2['event'] == ''
    finally:
        srv.shutdown()
```

- [ ] **Step 2: Run the test, verify it fails**

Run (Pi, in `~/embodied_ws`, env active): `python -m pytest src/embodied_mvp/tests/test_command_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embodied_mvp.command_server'`

- [ ] **Step 3: Write `src/embodied_mvp/embodied_mvp/command_server.py`**

```python
"""Thread-safe HTTP command/status server for agent_node.

POST /command  -- JSON body stored as the latest pending command
GET  /status   -- returns {"mode": ..., "event": ...}; a non-empty event is
                  one-shot, cleared after the first GET that returns it.

agent_node polls take_command() each control tick, calls set_mode() every
tick, and calls post_event() once when a notable event occurs.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CommandServer:
    def __init__(self, port):
        self._lock = threading.Lock()
        self._pending = None
        self._mode = 'IDLE'
        self._event = ''
        self._server = ThreadingHTTPServer(('0.0.0.0', int(port)),
                                           self._make_handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def take_command(self):
        with self._lock:
            cmd = self._pending
            self._pending = None
            return cmd

    def set_mode(self, mode):
        with self._lock:
            self._mode = mode

    def post_event(self, event):
        with self._lock:
            self._event = event

    def shutdown(self):
        try:
            self._server.shutdown()
        except Exception:
            pass

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not self.path.startswith('/command'):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    cmd = json.loads(self.rfile.read(n))
                    with server._lock:
                        server._pending = cmd
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                except Exception as e:  # noqa: BLE001
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(e).encode())

            def do_GET(self):
                if not self.path.startswith('/status'):
                    self.send_response(404)
                    self.end_headers()
                    return
                with server._lock:
                    body = json.dumps(
                        {'mode': server._mode, 'event': server._event}).encode()
                    server._event = ''   # one-shot
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `python -m pytest src/embodied_mvp/tests/test_command_server.py -v`
Expected: PASS — both tests green.

- [ ] **Step 5: Commit**

```bash
git add src/embodied_mvp/embodied_mvp/command_server.py src/embodied_mvp/tests/test_command_server.py
git commit -m "feat: add HTTP command/status server for agent_node"
```

---

## Task 3: Rename search_node -> agent_node (pure rename, no behavior change)

**Files:**
- Rename: `src/embodied_mvp/embodied_mvp/search_node.py` -> `agent_node.py`
- Modify: `src/embodied_mvp/setup.py`, `src/embodied_mvp/launch/mvp.launch.py`, `src/embodied_mvp/config/params.yaml`

- [ ] **Step 1: Git-rename the file**

Run: `git mv src/embodied_mvp/embodied_mvp/search_node.py src/embodied_mvp/embodied_mvp/agent_node.py`

- [ ] **Step 2: Rename the class and node name inside `agent_node.py`**

In `agent_node.py`: change `class SearchNode(Node):` -> `class AgentNode(Node):`,
`super().__init__('search_node')` -> `super().__init__('agent_node')`,
and in `main()` `node = SearchNode()` -> `node = AgentNode()`.

- [ ] **Step 3: Update `setup.py` entry point**

In `src/embodied_mvp/setup.py`, change the console_scripts line
`'search_node = embodied_mvp.search_node:main',`
to
`'agent_node = embodied_mvp.agent_node:main',`

- [ ] **Step 4: Update `launch/mvp.launch.py`**

In `src/embodied_mvp/launch/mvp.launch.py`, change the search node entry
`executable='search_node', name='search_node'`
to
`executable='agent_node', name='agent_node'`

- [ ] **Step 5: Update `params.yaml` section key**

In `src/embodied_mvp/config/params.yaml`, rename the top-level key
`search_node:` to `agent_node:` (params underneath unchanged).

- [ ] **Step 6: Build and regression-test**

Run (Pi, `~/embodied_ws`):
```bash
colcon build && source install/setup.bash
ros2 pkg executables embodied_mvp
```
Expected: lists `agent_node` (no `search_node`).
Then `ros2 launch embodied_mvp mvp.launch.py target_class:=bottle` — the
autonomous search still runs exactly as before (node now named `/agent_node`).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename search_node to agent_node"
```

---

## Task 4: Mode FSM + command integration in agent_node

**Files:**
- Modify: `src/embodied_mvp/embodied_mvp/agent_node.py`
- Modify: `src/embodied_mvp/config/params.yaml`

- [ ] **Step 1: Add new params (in `AgentNode.__init__`, with the other `declare_parameter` calls)**

```python
        self.declare_parameter('command_port', 9091)
        self.declare_parameter('manual_step_sec', 1.0)
        self.declare_parameter('manual_fwd_speed', 0.15)
        self.declare_parameter('manual_strafe_speed', 0.15)
        self.declare_parameter('manual_yaw_speed', 0.5)
        self.declare_parameter('rotate_90_sec', 1.4)   # calibrate empirically
        self.declare_parameter('photo_dwell_sec', 2.0)
```

And read them (with the other `p('...')` reads):

```python
        self.manual_step_sec = p('manual_step_sec').value
        self.manual_fwd = p('manual_fwd_speed').value
        self.manual_strafe = p('manual_strafe_speed').value
        self.manual_yaw = p('manual_yaw_speed').value
        self.rotate_90_sec = p('rotate_90_sec').value
        self.photo_dwell_sec = p('photo_dwell_sec').value
```

- [ ] **Step 2: Create the CommandServer and mode state (end of `__init__`)**

```python
        from embodied_mvp.command_server import CommandServer
        self.mode = 'IDLE'
        self.manual_twist = (0.0, 0.0, 0.0)   # vx, vy, wz
        self.manual_deadline = 0.0
        self.rp_index = 0
        self.rp_phase = 'dwell'
        self.rp_phase_start = 0.0
        self.cmd_server = CommandServer(self.get_parameter('command_port').value)
        self.get_logger().info(
            f"agent_node command server on :{self.get_parameter('command_port').value}")
```

- [ ] **Step 3: Add a full-twist publisher (next to `publish_cmd`)**

The existing `publish_cmd(vx, wz)` cannot strafe. Add:

```python
    def publish_full_cmd(self, vx, vy, wz):
        t = Twist()
        t.linear.x = vx
        t.linear.y = vy
        t.angular.z = wz
        self.cmd_pub.publish(t)
```

- [ ] **Step 4: Add the command-to-mode handler**

```python
    def _dir_to_twist(self, direction):
        f, s, y = self.manual_fwd, self.manual_strafe, self.manual_yaw
        return {
            'forward':      (f,  0.0, 0.0),
            'backward':     (-f, 0.0, 0.0),
            'left':         (0.0,  s, 0.0),
            'right':        (0.0, -s, 0.0),
            'rotate_left':  (0.0, 0.0,  y),
            'rotate_right': (0.0, 0.0, -y),
        }.get(direction, (0.0, 0.0, 0.0))

    def apply_command(self, cmd, now):
        action = cmd.get('action')
        if action == 'stop':
            self.mode = 'IDLE'
        elif action == 'move':
            self.mode = 'MANUAL'
            self.manual_twist = self._dir_to_twist(cmd.get('dir'))
            self.manual_deadline = now + self.manual_step_sec
        elif action == 'find':
            target = cmd.get('target')
            if target:
                self.target_class = target
                self.reset_search(now)
                self.mode = 'SEARCH'
        elif action == 'rotate_photo':
            self.mode = 'ROTATE_PHOTO'
            self.rp_index = 0
            self.rp_phase = 'dwell'
            self.rp_phase_start = now
            self.cmd_server.post_event('photo_ready:1')
        self.get_logger().info(f'command {cmd} -> mode {self.mode}')
```

- [ ] **Step 5: Add `reset_search` (resets the SEARCH sub-state)**

Collect the existing SEARCH-state initialisers into one method so a `find`
command can restart a clean search:

```python
    def reset_search(self, now):
        self.state = 'SEARCHING'
        self.last_target = None
        self.last_target_time = 0.0
        self.start_time = now
        self.consec_hits = 0
        self.scan_phase = 'rotate'
        self.scan_phase_start = now
        self.scan_dir = 1.0
        self.scan_bursts = 0
        self.reacquire_until = 0.0
        self.burst_end = 0.0
        self.last_acted_time = 0.0
```

- [ ] **Step 6: Rename the existing `tick` body to `tick_search`**

Rename the current `def tick(self):` method to `def tick_search(self):`.
Inside it, when the search reaches the ARRIVED outcome, replace the line that
sets `self.state = 'ARRIVED'` (in the arrival branch) with:

```python
                self.cmd_server.post_event(f'arrived:{self.target_class}')
                self.mode = 'IDLE'
                self.state = 'SEARCHING'
                return
```

Also delete the now-obsolete `if self.state == 'ARRIVED':` early-return block
(arrival now switches mode to IDLE instead of latching an ARRIVED state).

- [ ] **Step 7: Add `tick_manual` and `tick_rotate_photo`**

```python
    def tick_manual(self, now):
        if now >= self.manual_deadline:
            self.mode = 'IDLE'
            self.publish_full_cmd(0.0, 0.0, 0.0)
            return
        vx, vy, wz = self.manual_twist
        self.publish_full_cmd(vx, vy, wz)

    def tick_rotate_photo(self, now):
        phase_t = now - self.rp_phase_start
        if self.rp_phase == 'dwell':
            self.publish_full_cmd(0.0, 0.0, 0.0)   # still -> sharp photo
            if phase_t >= self.photo_dwell_sec:
                self.rp_phase = 'rotate'
                self.rp_phase_start = now
        else:  # rotate ~90 deg
            self.publish_full_cmd(0.0, 0.0, self.manual_yaw)
            if phase_t >= self.rotate_90_sec:
                self.rp_index += 1
                if self.rp_index >= 4:
                    self.mode = 'IDLE'
                    self.cmd_server.post_event('rotate_photo_done')
                else:
                    self.rp_phase = 'dwell'
                    self.rp_phase_start = now
                    self.cmd_server.post_event(f'photo_ready:{self.rp_index + 1}')
```

- [ ] **Step 8: Add the new top-level `tick` that dispatches by mode**

```python
    def tick(self):
        now = time.time()
        cmd = self.cmd_server.take_command()
        if cmd is not None:
            self.apply_command(cmd, now)

        if self.mode == 'IDLE':
            self.publish_full_cmd(0.0, 0.0, 0.0)
            self.publish_pantilt(0.0, 0.0)
        elif self.mode == 'MANUAL':
            self.tick_manual(now)
        elif self.mode == 'SEARCH':
            self.tick_search()
        elif self.mode == 'ROTATE_PHOTO':
            self.tick_rotate_photo(now)

        self.cmd_server.set_mode(self.mode)
```

- [ ] **Step 9: Add CommandServer shutdown to `destroy_node` (or `main` finally)**

In `main()`'s `finally` block, before `node.destroy_node()`, add:
`node.cmd_server.shutdown()`

- [ ] **Step 10: Add the new params to `params.yaml`**

Under the `agent_node:` `ros__parameters:` block, add:

```yaml
    command_port: 9091
    manual_step_sec: 1.0
    manual_fwd_speed: 0.15
    manual_strafe_speed: 0.15
    manual_yaw_speed: 0.5
    rotate_90_sec: 1.4            # calibrate: time for a ~90 deg in-place turn
    photo_dwell_sec: 2.0
```

- [ ] **Step 11: Build**

Run (Pi): `colcon build && source install/setup.bash`
Expected: build succeeds, no errors.

- [ ] **Step 12: Verify mode switching with curl (car wheels off the ground)**

Start: `ros2 launch embodied_mvp mvp.launch.py`
In another terminal:
```bash
curl -s -X POST localhost:9091/command -d '{"action":"move","dir":"forward"}'
curl -s localhost:9091/status
```
Expected: status shows `"mode": "MANUAL"`, wheels spin forward ~1 s then stop;
status returns to `"mode": "IDLE"`.
```bash
curl -s -X POST localhost:9091/command -d '{"action":"rotate_photo"}'
curl -s localhost:9091/status
```
Expected: `mode` cycles ROTATE_PHOTO; `event` returns `photo_ready:1` once,
then later `photo_ready:2`..`4` and finally `rotate_photo_done`; car does 4
turns.
```bash
curl -s -X POST localhost:9091/command -d '{"action":"find","target":"bottle"}'
```
Expected: `mode` becomes SEARCH; sending any other command immediately
switches mode (interrupt).

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat: add IDLE/MANUAL/SEARCH/ROTATE_PHOTO modes to agent_node"
```

---

## Task 5: Telegram bot (laptop)

**Files:**
- Create: `laptop/telegram_bot.py`

- [ ] **Step 1: Install the Telegram library**

Run (laptop): `pip install python-telegram-bot`
Expected: `Successfully installed python-telegram-bot-...` (v21+).

- [ ] **Step 2: Create a Telegram bot and get a token**

On the phone: open Telegram, message `@BotFather`, send `/newbot`, follow
prompts, copy the HTTP API token. Also message your new bot once, then open
`https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser to read your
numeric `from.id` — that is the authorized user ID.

- [ ] **Step 3: Write `laptop/telegram_bot.py`**

```python
"""Telegram bot — phone natural-language control for the search robot.

Receives messages, keyword-parses them (nl_parser), POSTs commands to the Pi
agent_node, fetches camera snapshots, and polls agent status to relay arrival
and rotate-photo events back to the phone.

Setup:
    pip install python-telegram-bot requests
Config: set BOT_TOKEN, PI_IP, AUTHORIZED_IDS below.
Run:    python telegram_bot.py
"""
import io
import threading
import time

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from nl_parser import parse, SUPPORTED_TARGETS_CN

# ---- config ----
BOT_TOKEN = 'PUT-YOUR-BOTFATHER-TOKEN-HERE'
PI_IP = '192.168.178.37'
AUTHORIZED_IDS = {123456789}        # your Telegram numeric user id(s)
# ----------------

CMD_URL = f'http://{PI_IP}:9091/command'
STATUS_URL = f'http://{PI_IP}:9091/status'
SNAPSHOT_URL = f'http://{PI_IP}:8080/snapshot'

HELP = ('支持指令：前进/后退/左移/右移/左转/右转/停/拍照/旋转拍照/'
        '去找<目标>。目标：' + SUPPORTED_TARGETS_CN)

PHOTO_LABELS = {1: '前', 2: '右', 3: '后', 4: '左'}

_session = requests.Session()


def post_command(cmd):
    """POST a command to the Pi. Returns True on success."""
    try:
        _session.post(CMD_URL, json=cmd, timeout=2.0)
        return True
    except requests.RequestException:
        return False


def fetch_snapshot():
    """GET the current camera frame. Returns JPEG bytes or None."""
    try:
        r = _session.get(SNAPSHOT_URL, timeout=3.0)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def get_status():
    try:
        return _session.get(STATUS_URL, timeout=2.0).json()
    except requests.RequestException:
        return None


async def _send_photo(context, chat_id, caption):
    jpg = fetch_snapshot()
    if jpg is None:
        await context.bot.send_message(chat_id, '取图失败')
        return
    await context.bot.send_photo(chat_id, photo=io.BytesIO(jpg), caption=caption)


def _poll_events(loop, context, chat_id, stop_after_sec=120):
    """Background thread: poll Pi status, relay arrival / rotate-photo events."""
    import asyncio
    deadline = time.time() + stop_after_sec
    while time.time() < deadline:
        time.sleep(0.7)
        st = get_status()
        if st is None:
            continue
        event = st.get('event', '')
        if not event:
            continue
        if event.startswith('arrived:'):
            target = event.split(':', 1)[1]
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id, f'已到达 {target}'), loop)
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id, f'到达 {target}'), loop)
            return
        if event.startswith('photo_ready:'):
            n = int(event.split(':', 1)[1])
            label = PHOTO_LABELS.get(n, str(n))
            asyncio.run_coroutine_threadsafe(
                _send_photo(context, chat_id, f'旋转拍照 {n}/4 ({label})'), loop)
        elif event == 'rotate_photo_done':
            asyncio.run_coroutine_threadsafe(
                context.bot.send_message(chat_id, '旋转拍照完成'), loop)
            return


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in AUTHORIZED_IDS:
        return  # ignore unauthorized users
    chat_id = update.effective_chat.id
    text = update.message.text or ''
    cmd = parse(text)

    if cmd is None:
        await context.bot.send_message(chat_id, '没听懂。' + HELP)
        return

    if cmd['action'] == 'photo':
        await _send_photo(context, chat_id, '当前画面')
        return

    if cmd['action'] == 'find' and cmd['target'] is None:
        await context.bot.send_message(chat_id, '不认识该目标。' + HELP)
        return

    if not post_command(cmd):
        await context.bot.send_message(chat_id, '小车失联')
        return

    if cmd['action'] == 'find':
        await context.bot.send_message(chat_id, f"前往寻找 {cmd['target']} ...")
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events, args=(loop, context, chat_id),
                         daemon=True).start()
    elif cmd['action'] == 'rotate_photo':
        await context.bot.send_message(chat_id, '开始旋转拍照 ...')
        loop = __import__('asyncio').get_running_loop()
        threading.Thread(target=_poll_events, args=(loop, context, chat_id),
                         daemon=True).start()
    else:
        await context.bot.send_message(chat_id, f"执行：{text}")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    print('telegram bot running — message your bot from the phone')
    app.run_polling()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verify the parser import and dry behavior (no robot needed)**

Run (laptop, `laptop/`): `python -c "from telegram_bot import HELP; print(HELP)"`
Expected: prints the help string with the command list — confirms imports
resolve (`nl_parser`, `telegram`, `requests` all importable).

- [ ] **Step 5: Commit**

```bash
git add laptop/telegram_bot.py
git commit -m "feat: add Telegram bot for phone control"
```

---

## Task 6: Launch wiring + README

**Files:**
- Modify: `src/embodied_mvp/launch/mvp.launch.py`
- Modify: `README.md`

- [ ] **Step 1: Confirm agent_node starts in the launch file**

Verify `mvp.launch.py` already launches `agent_node` (renamed in Task 3,
Step 4). No `enable_search` gating is needed — `agent_node` simply starts in
IDLE and waits for commands. If the node still has a `condition=` from the old
`search_node`, remove it so the node always runs.

- [ ] **Step 2: Add a phone-control section to `README.md`**

Append to `README.md`:

```markdown
## Phone control (Telegram)

Run YOLO offload as usual (`laptop_detector.py`), plus the bot:

1. Create a bot via @BotFather, get the token.
2. Edit `laptop/telegram_bot.py` — set `BOT_TOKEN`, `PI_IP`, `AUTHORIZED_IDS`.
3. `pip install python-telegram-bot requests`
4. `python laptop/telegram_bot.py`

Commands (Chinese): 前进 / 后退 / 左移 / 右移 / 左转 / 右转 / 停 /
拍照 / 旋转拍照 / 去找<目标>（瓶子、椅子、人 ...）.

`agent_node` on the Pi receives commands on port 9091 and is the sole
/cmd_vel owner (modes: IDLE / MANUAL / SEARCH / ROTATE_PHOTO).
```

- [ ] **Step 3: Commit**

```bash
git add src/embodied_mvp/launch/mvp.launch.py README.md
git commit -m "docs: add phone-control launch wiring and README section"
```

---

## Task 7: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Calibrate `rotate_90_sec`**

Car wheels on the ground. `ros2 launch embodied_mvp mvp.launch.py`.
`curl -s -X POST localhost:9091/command -d '{"action":"rotate_photo"}'`.
Watch one quarter-turn. If it over/under-shoots 90 deg, edit `rotate_90_sec`
in `params.yaml`, `colcon build`, retry until each turn is ~90 deg and four
turns end roughly at the start heading.

- [ ] **Step 2: Full bot run**

Pi: `ros2 launch embodied_mvp mvp.launch.py`.
Laptop: `python laptop/laptop_detector.py --pi <PI_IP>` and
`python laptop/telegram_bot.py`.

- [ ] **Step 3: Exercise every command from the phone**

- "前进" → car steps forward ~1 s, stops.
- "左转" → car rotates left briefly.
- "拍照" → a photo arrives in Telegram.
- "旋转拍照" → 4 labelled photos (前/右/后/左) arrive, "旋转拍照完成" sent,
  car ends ≈ original heading.
- "去找瓶子" (bottle in the room) → car searches, approaches, "已到达 bottle"
  + photo arrive.
- Send "停" mid-search → car stops immediately (interrupt).
- Send a nonsense message → bot replies with the help text.

- [ ] **Step 4: Regression — autonomous search still works**

The original flow is unchanged: `agent_node` idles until a `find` command;
the SEARCH behaviour is the former `search_node` logic. Confirm a `find`
command reproduces the pre-existing search/approach/arrival behaviour.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "test: end-to-end phone control verified"
```

---

## Self-Review Notes

- **Spec coverage:** Telegram bot (Task 5), keyword parsing (Task 1), bot on
  laptop (Task 5), fixed-step motion (Task 4 `tick_manual`), interrupt (Task 4
  `apply_command` overwrites `mode`), arrival notify+photo (Task 4 Step 6 +
  Task 5 `_poll_events`), rotate-photo (Task 4 `tick_rotate_photo` + Task 5).
  All spec sections map to a task.
- **No placeholders:** every code step has complete code. `BOT_TOKEN` /
  `PI_IP` / `AUTHORIZED_IDS` are real config values the user fills per their
  account — flagged explicitly in Task 5 Steps 2-3, not plan placeholders.
- **Type consistency:** command dict shape from `parse()` (Task 1) matches the
  keys read in `apply_command()` (Task 4) — `action`, `dir`, `target`. Status
  dict `{mode, event}` consistent between `CommandServer` (Task 2) and the bot
  poller (Task 5). Event strings `arrived:<x>`, `photo_ready:<n>`,
  `rotate_photo_done` consistent across Tasks 4 and 5.
