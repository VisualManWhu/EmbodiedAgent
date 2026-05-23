# EmbodiedAgent — Pi5 Mecanum Search-and-Map Robot

A Raspberry Pi 5 mecanum-wheel car that:

1. **Searches and approaches** COCO objects via camera → YOLO → visual servo (MVP).
2. **Listens for natural-language commands** from a phone over Telegram
   (motion, photo, rotate-photo, find-by-class).
3. **Builds a semantic map** of the room — AprilTag-anchored localization
   plus multi-view-fused object landmarks (semantic SLAM phase 1).

Detection runs on a laptop GPU and POSTs results back to the Pi, so the Pi's
CPU/battery BMS stays light. See [`docs/CONTEXT.md`](docs/CONTEXT.md) for the
full background, gotchas, and decisions log.

## Architecture

```
Phone (Telegram)
  | Telegram
Laptop (Windows + RTX 4060):
  telegram_bot.py    — keyword parser, posts commands to Pi
  laptop_detector.py — pulls Pi /snapshot, runs YOLOv8x + AprilTag,
                       fuses semantic map, serves map.png :8091
  | HTTP (LAN)
Pi5 (ROS2 Humble, RoboStack conda env `ros_env`):
  csi_camera_node    — spawns rpicam-vid → /camera/image_raw
  motor_node         — mecanum kinematics; loborobot backend
  ir_node            — HC-SR04 ultrasonic → /ir/range
  side_ir_node       — binary side IR → /ir/left /ir/right
  pantilt_node       — pan-tilt servos via PCA9685
  det_bridge_node    — receives laptop detections (:9090) → /detections
  mjpeg_node         — MJPEG stream + /snapshot (:8080); reports pan-tilt
                       in /snapshot response headers
  agent_node         — mode FSM (IDLE/MANUAL/SEARCH/ROTATE_PHOTO),
                       sole /cmd_vel owner, command HTTP server :9091
  semantic_map_node  — receives map JSON (:9092), publishes MarkerArray + TF
  yolo_node          — on-Pi YOLO (OFF by default; detection is offloaded)
```

Hardware-touching nodes (`motor_node`, `ir_node`, `pantilt_node`, etc.) all
have a `mock` backend so the stack starts without hardware. Switch to real
backends in [`src/embodied_mvp/config/params.yaml`](src/embodied_mvp/config/params.yaml).

## Repo layout

- [`src/embodied_mvp/`](src/embodied_mvp/) — ROS2 Humble package (Pi nodes, launch, params).
- [`laptop/`](laptop/) — laptop-side scripts.
  - [`laptop_detector.py`](laptop/laptop_detector.py) — YOLO + semantic SLAM.
  - [`telegram_bot.py`](laptop/telegram_bot.py) — Telegram NL control.
  - [`slam/`](laptop/slam/) — calibration, AprilTag detection, pose estimator,
    semantic map, map renderer, printable target generator.
- [`docs/CONTEXT.md`](docs/CONTEXT.md) — context + gotchas for picking up work.
- [`docs/superpowers/`](docs/superpowers/) — design specs and plans.

## Pi5 first-time setup

```bash
# 1. ROS2 Humble via RoboStack conda (avoids Bookworm apt-pin gaps)
curl -L https://micro.mamba.pm/install.sh | bash
mamba create -n ros_env python=3.11 -c conda-forge -y
mamba activate ros_env
mamba install ros-humble-desktop ros-humble-v4l2-camera \
              ros-humble-vision-msgs ros-humble-visualization-msgs \
              ros-humble-tf2-ros ros-humble-cv-bridge \
              ros-humble-teleop-twist-keyboard \
              compilers cmake pkg-config -c robostack-staging -y

# 2. Python deps for kit drivers (when leaving `mock` backends)
pip install pyserial adafruit-circuitpython-servokit smbus2

# 3. Workspace
mkdir -p ~/embodied_ws/src
# copy this repo's src/embodied_mvp into ~/embodied_ws/src/
cd ~/embodied_ws
colcon build --packages-select embodied_mvp
source install/setup.bash
```

Each new shell on the Pi:
```bash
conda activate ros_env
source ~/embodied_ws/install/setup.bash
```

## Laptop setup (Windows, Python 3.11, RTX 4060)

```powershell
pip install ultralytics opencv-python requests numpy pyyaml python-telegram-bot
pip install pupil-apriltags          # only needed for --slam
# CUDA build of PyTorch (replace cu124 to match your driver):
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Telegram bot config (one-time):
```powershell
cd laptop
Copy-Item telegram_bot_config.example.txt telegram_bot_config.txt
# Edit telegram_bot_config.txt — BOT_TOKEN, PI_IP, AUTHORIZED_IDS.
# This file is gitignored; never commit it.
```

## Run

### MVP autonomous search only

```bash
# Pi
ros2 launch embodied_mvp mvp.launch.py target_class:=chair
```
```powershell
# Laptop (separate machine)
cd laptop
python laptop_detector.py --pi 172.20.10.4
```

### Add phone control

```powershell
# Laptop, additional terminal
cd laptop
python telegram_bot.py
```

Then message your bot from the phone. Commands (Chinese / English):

- Motion: `前进` / `后退` / `左移` / `右移` / `左转` / `右转` / `停`
  - Optional duration: `前进3秒`, `back 5s` (capped at 30 s).
- Photo: `拍照`, `旋转拍照`.
- Search: `去找椅子`, `find bottle` — `agent_node` switches to SEARCH and reports `arrived:<class>` when stopped.
- Map: `地图` — top-down semantic map PNG (requires `--slam`).

### Add semantic SLAM

```bash
# Pi — same launch with the map node enabled
ros2 launch embodied_mvp mvp.launch.py enable_semantic_map:=true
```
```powershell
# Laptop — detector with SLAM, recommended tuning
python laptop_detector.py --pi 172.20.10.4 --slam `
  --cam-height 0.15 `
  --conf 0.5 `
  --gate-radius 0.8 `
  --tentative-stale-sec 60 `
  --confirm-min-obs 3 `
  --miss-prune 12
```
`--cam-height` is the camera's height above the floor (m) with pan-tilt at
zero — measure it; the default 0.15 matches this kit. Ground-plane object
placement is sensitive to this value.

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

### View the map

- Browser: <http://127.0.0.1:8091/map.png> (laptop) — top-down PNG.
- Telegram: send `地图`.
- RViz2 (Pi desktop or VNC): Fixed Frame `map`; subscribe `MarkerArray`
  topic `/semantic_map/markers`.
- Pi camera live: <http://172.20.10.4:8080/> (MJPEG).
- Annotated YOLO frame: <http://127.0.0.1:8090/annotated>.

### Stop everything

`Ctrl+C` in each window. Press `q` in the laptop detector preview.

## Semantic SLAM bring-up (one-time room prep)

1. **Generate printable targets:**
   ```powershell
   cd laptop
   python -m slam.make_targets --tags 0-7
   ```
   PNGs land in `laptop/slam/print_targets/`.

2. **Camera calibration** (run with the Pi camera stack up):
   ```powershell
   # capture 15-20 chessboard shots
   python -m slam.camera_calib capture --url http://172.20.10.4:8080/snapshot --out calib_shots
   # compute intrinsics, dropping bad views
   python -m slam.camera_calib calibrate --images calib_shots --square-m <measured_square_size> --max-view-error 0.8
   ```
   Target RMS < 0.5 px. Output overwrites
   [`laptop/slam/camera_intrinsics.yaml`](laptop/slam/camera_intrinsics.yaml).

3. **Place + measure AprilTags:**
   - Print `tag36h11_id0..N.png`, mount on hard board, fix to walls.
   - Tags must not move afterwards.
   - Measure each tag's `x, y, z, yaw_deg, size_m`. Tag 0 is the map origin.
   - Write the real values to `laptop/slam/tag_map.local.yaml` (gitignored).
     The tracked [`tag_map.yaml`](laptop/slam/tag_map.yaml) is a template only;
     `load_tag_map()` automatically prefers `.local` if present.

## Topic map

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/image_raw` | sensor_msgs/Image | csi_camera → yolo / mjpeg |
| `/detections` | vision_msgs/Detection2DArray | det_bridge → agent |
| `/ir/range` | sensor_msgs/Range | ir_node → agent |
| `/ir/left`, `/ir/right` | std_msgs/Bool | side_ir → agent |
| `/cmd_vel` | geometry_msgs/Twist | agent → motor |
| `/pantilt/cmd` | geometry_msgs/Vector3 | agent → pantilt / mjpeg headers |
| `/semantic_map/markers` | visualization_msgs/MarkerArray | semantic_map_node → RViz |
| TF `map → base_link → camera` | tf2 | semantic_map_node |

## HTTP endpoints

| URL | Server | Purpose |
|-----|--------|---------|
| `http://<pi>:8080/` | `mjpeg_node` | MJPEG stream (browser) |
| `http://<pi>:8080/snapshot` | `mjpeg_node` | latest JPEG + `X-Pan-Yaw` / `X-Pan-Tilt` headers |
| `http://<pi>:9090/detections` | `det_bridge_node` | POST detections from laptop |
| `http://<pi>:9091/command` | `agent_node` | POST commands from telegram_bot |
| `http://<pi>:9091/status` | `agent_node` | GET mode / event (telegram_bot polls) |
| `http://<pi>:9092/map` | `semantic_map_node` | POST semantic map from laptop |
| `http://<laptop>:8090/annotated` | `laptop_detector.py` | latest YOLO-boxed JPEG |
| `http://<laptop>:8091/map.png` | `laptop_detector.py --slam` | top-down semantic map PNG |
| `http://<laptop>:8092/nav/*` | `laptop_detector.py --slam` | nav api for telegram_bot (candidates, goto, done, landmarks, tags) |

## Hardware bring-up checklist

Confirm before flipping any backend from `mock` to real in `params.yaml`:

| Item | Find | Param |
|------|------|-------|
| Motor protocol | UART / I2C / PCA9685 + GPIO direction | `motor_node.backend`, `serial_port` |
| Wheelbase | half-distances (m) | `wheel_base_lx`, `wheel_base_ly` |
| Ultrasonic pins | GPIO trigger / echo | `ir_node.trigger_pin`, `echo_pin` |
| Side IR pins | left / right GPIO | `side_ir_node.left_pin`, `right_pin` |
| Servo board | PCA9685 I2C address | `pantilt_node.i2c_address` |
| Servo channels | pan / tilt PWM channels | `pantilt_node.pan_channel`, `tilt_channel` |
| Camera height | measure mount to floor (m) | `laptop_detector.py --cam-height` |

## NCNN speedup (Pi-side YOLO fallback)

When the laptop is unavailable and you must run YOLO on the Pi:

On the laptop:
```bash
yolo export model=yolov8n.pt format=ncnn imgsz=320 int8=True
# scp yolov8n_ncnn_model/ pi@<pi-ip>:~/embodied_ws/
```
In Pi `params.yaml`:
```yaml
yolo_node:
  ros__parameters:
    backend: ncnn
    model_path: /home/pi/embodied_ws/yolov8n_ncnn_model
```
Then `enable_yolo:=true enable_bridge:=false` in the launch.
Warning: on-Pi YOLO can pin the CPU and trip the motor-battery BMS — keep
detection on the laptop in normal operation.

## End-to-end verification (room test)

1. Place a chair 2-3 m from the car.
2. Pi: `ros2 launch embodied_mvp mvp.launch.py target_class:=chair`
3. Laptop: `python laptop_detector.py --pi 172.20.10.4`
4. Expect: rotate → spot chair → discrete forward pulses, recentering → bbox
   height ≥ `arrived_height_ratio` (or IR < `stop_distance_m`) → stop, log
   `Found chair!`, Telegram receives `arrived:chair`.

For mapping verification see the semantic-SLAM phase plan in
[`docs/CONTEXT.md`](docs/CONTEXT.md).

## Security notes

- `laptop/telegram_bot_config.txt` is gitignored — keep the bot token,
  authorized user IDs, and Pi IP there. **Never commit them.**
- `laptop/slam/tag_map.local.yaml` is gitignored — keep real room geometry
  there. The tracked `tag_map.yaml` is a template only.
- The Pi HTTP endpoints (`:9090`, `:9091`, `:9092`) are unauthenticated;
  do not expose the Pi to untrusted networks.

## TODO

- **Auto-built tag pose graph.** Today tag positions are hand-measured.
  Auto-solving the tag layout from co-visible observations would remove
  the manual measurement step at the cost of some drift.
