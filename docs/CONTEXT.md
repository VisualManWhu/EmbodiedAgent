# Project Context — EmbodiedAgent

Continuation notes for picking up work in a fresh session. Read this first,
then `README.md`, then the relevant files in `docs/superpowers/`.

## What this project is

An autonomous object-search robot car. A Raspberry Pi 5 mecanum-wheel car finds
and approaches COCO-class objects, controllable by natural-language messages
from a phone. Two phases are complete and merged to `main`:

1. **MVP autonomous search** — camera → YOLO detection → visual-servo search →
   mecanum motion, with ultrasonic stop and (unwired) side-IR sensors.
2. **Phone natural-language control** — Telegram bot: basic motion, photo,
   rotate-and-photo, on-the-fly COCO-80 target search.

Next phase (not started): **semantic SLAM** — see the section below.

## Hardware

- **Raspberry Pi 5** (4 GB), Raspbian Bookworm. On home WiFi, IP `192.168.178.12`.
- **Makerobo (创乐博) mecanum 4WD kit.** Vendor control library `LOBOROBOT`
  (vendored + smbus→smbus2-patched as `loborobot_lib.py`).
  - 4 motors via PCA9685 (I2C `0x40`); motor-D direction also uses GPIO 24/25.
  - Mecanum wheel→motor index: FL=0, FR=1, RL=2, RR=3.
- **CSI camera** OV5647 (Pi Camera v1 sensor) on a 2-DOF pan-tilt.
  Servos via PCA9685: channel 10 = pan (centre 90°), channel 9 = tilt (centre -10°).
- **HC-SR04 ultrasonic** front distance: GPIO trigger 20, echo 21.
- **Binary IR** side obstacle sensors: GPIO 12 (left), 16 (right).
- **No wheel encoders. No IMU.** All motion is open-loop.
- **4000 mAh battery**, shared by Pi5 + motors. Its BMS trips under combined load.
- **Laptop**: Windows, Python 3.11, RTX 4060 (CUDA 12.9). 500 Mb home broadband.

## Architecture (current)

Detection is **offloaded to the laptop GPU** — running YOLO on the Pi pegged its
CPU, the current spike tripped the battery BMS, and the Pi shut down. With YOLO
on the laptop the Pi5 CPU stays light and the BMS holds.

```
Phone (Telegram app)
  | Telegram
Laptop:  telegram_bot.py   — bot + keyword parser (nl_parser.py)
         laptop_detector.py — reads Pi /snapshot, YOLOv8x on GPU, posts
                              detections back, serves annotated frames /annotated:8090
  | HTTP (LAN)
Pi5 (ROS2 Humble, RoboStack conda env `ros_env`):
  csi_camera_node   — spawns system `rpicam-vid` (MJPEG) → /camera/image_raw
  motor_node        — mecanum kinematics + slew limiting; loborobot backend
  ir_node           — HC-SR04 ultrasonic → /ir/range
  side_ir_node      — binary side IR → /ir/left /ir/right
  pantilt_node      — servos via PCA9685
  det_bridge_node   — receives laptop detections (HTTP :9090) → /detections
  mjpeg_node        — MJPEG stream + /snapshot endpoint (:8080)
  agent_node        — mode FSM, sole /cmd_vel owner, HTTP command server :9091
  yolo_node         — on-Pi YOLO (default OFF; detection is offloaded)
```

`agent_node` modes: `IDLE` / `MANUAL` (timed step) / `SEARCH` (autonomous
search FSM) / `ROTATE_PHOTO` (4× turn-and-photo). A new command instantly
switches mode = the interrupt mechanism. `command_server.py` is the HTTP helper.

## Key decisions & gotchas (hard-won — do not re-learn these)

- **Power / BMS:** Pi5 + 4 motors on one 4000 mAh pack. YOLO on the Pi → CPU
  100% → +~2 A → BMS overcurrent trip → hard shutdown. Fixed by offloading
  detection to the laptop. Pi5 independent power (separate bank) was considered
  but not done (rewiring risk). Motor params kept conservative
  (`max_move_percent` 45, slew-limited) to cap current.
- **conda libcamera can't see the Pi5 CSI camera** (missing PiSP IPA modules).
  `csi_camera_node` spawns the SYSTEM `rpicam-vid` binary as a subprocess.
- **Subprocess env:** the spawned `rpicam-vid` must run with conda paths
  stripped from `LD_LIBRARY_PATH`, or it loads conda libs and crashes.
- **Camera codec:** use `rpicam-vid --codec mjpeg` + `cv2.imdecode` — raw YUV420
  plane-order (I420 vs YV12) caused a colour cast.
- **Detection is sparse (~0.5–1 Hz) and latent (~1 s).** Continuous visual
  servoing on stale bboxes hunts/overshoots. Search therefore uses a
  time-windowed target lock + a discrete-pulse APPROACHING (one short move per
  fresh detection) + stop-and-scan search.
- **No odometry:** `rotate_photo` returns to the original heading only
  approximately (4× open-loop 90° turns).
- **Arrival is vision-based** (bbox height fraction), because the ultrasonic
  beam misses thin objects like bottles.
- **Telegram** is the phone channel; the operator has access sorted.
- ROS2 was installed via **RoboStack conda** (`ros_env`), not apt — Bookworm is
  not an official ROS2 target. `colcon build` non-symlink (setuptools issue).
- Build on Windows is dev-only; the Pi (`~/embodied_ws`) is the run target.
  Workflow: edit on Windows → scp to Pi → `colcon build` → run.

## Repo layout

- `src/embodied_mvp/` — ROS2 Humble package (nodes, launch, params).
- `laptop/` — laptop-side scripts (detector, bot, parser, tests).
- `docs/superpowers/specs/` — design specs (phone-control spec is here).
- `docs/superpowers/plans/` — implementation plans.
- `docs/CONTEXT.md` — this file.
- Original MVP plan: `C:\Users\yanfeng\.claude\plans\5-4g-...-garden.md`.

## Known TODOs

- **Side-IR obstacle avoidance is not wired.** `obs_left`/`obs_right` are read
  and published but no behavior uses them — the avoidance code was dropped in
  the discrete-pulse APPROACHING rewrite. See the README TODO. The robot only
  *appears* to avoid obstacles (visual servo curves the path incidentally).

## Next phase — Semantic SLAM

**Goal:** SLAM that produces a map annotated with semantic objects — labelled
COCO detections localized in a persistent map — **not just occupancy geometry**.
The end state is map-based global navigation that can reason about objects
("go to the chair in the kitchen"), building on the existing detection pipeline.

**Constraints / things to weigh during brainstorming:**

- **No wheel encoders, no IMU** → no metric odometry. Pure monocular visual
  SLAM is scale-ambiguous and drifts. The original MVP plan considered AprilTag
  anchors + RTAB-Map monocular. Adding an IMU (MPU6050) is a possible hardware
  step but the operator has so far avoided hardware changes.
- **The detection pipeline already exists** (laptop YOLOv8x, offloaded). The
  semantic layer should consume those detections and place them in the map.
- **Compute placement:** SLAM is heavy. Detection already runs on the laptop;
  SLAM most likely runs on the laptop too (Pi streams images, as it does now).
  Decide the Pi/laptop split deliberately.
- **Backend choice:** RTAB-Map vs ORB-SLAM3 (monocular) vs visual-inertial if an
  IMU is added; object-level / semantic mapping approach.
- **Navigation:** Nav2 vs a custom planner; how the semantic map feeds goals.
- **Coordinate frames:** map ↔ odom ↔ base_link ↔ camera, with no odom source.
- **Power budget:** keep the BMS lesson in mind — anything heavy stays off the Pi.

## Starting the next session

1. Open this repo; read `docs/CONTEXT.md`, `README.md`,
   `docs/superpowers/specs/`, `docs/superpowers/plans/`.
2. Brainstorm the semantic-SLAM feature (superpowers brainstorming flow):
   clarify scope, weigh the constraints above, propose approaches.
3. Spec → plan → implement as before.
