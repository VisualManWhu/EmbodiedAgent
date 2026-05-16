# EmbodiedAgent — Pi5 Mecanum Search-and-Approach MVP

ROS2 Humble package for a Raspberry Pi 5 mecanum-wheel car: detects a COCO object via USB camera and visually servos toward it, stopping on IR proximity.

Plan: `C:\Users\yanfeng\.claude\plans\5-4g-rgb-4000ma-raspbian-bookworm-rtx40-abundant-garden.md`

## Architecture

```
camera ──► yolo_node ──► /detections ──┐
                                       ├──► search_node ──► /cmd_vel ──► motor_node ──► wheels
ir_node ──► /ir/range  ────────────────┘                  └► /pantilt/cmd ──► pantilt_node
```

All hardware nodes (`motor_node`, `ir_node`, `pantilt_node`) ship with **`mock` backend default** so the stack runs without hardware for software validation. Switch to real backends in `config/params.yaml` once kit specifics are known.

## Pi5 first-time setup

```bash
# 1. ROS2 Humble (via RoboStack conda — avoids Bookworm apt-pin gaps)
curl -L https://micro.mamba.pm/install.sh | bash
mamba create -n ros_env python=3.11 -c conda-forge -y
mamba activate ros_env
mamba install ros-humble-desktop ros-humble-v4l2-camera \
              ros-humble-vision-msgs ros-humble-cv-bridge \
              ros-humble-teleop-twist-keyboard \
              compilers cmake pkg-config -c robostack-staging -y

# 2. Python deps
pip install ultralytics opencv-python numpy pyserial
# Servo (only if using pca9685 backend):
pip install adafruit-circuitpython-servokit

# 3. Workspace
mkdir -p ~/embodied_ws/src
cd ~/embodied_ws/src
# copy this repo's `src/embodied_mvp` here
cd ~/embodied_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
source ~/embodied_ws/install/setup.bash
ros2 launch embodied_mvp mvp.launch.py target_class:=chair
```

Stop motors only (debug detection without driving):
```bash
ros2 launch embodied_mvp mvp.launch.py enable_search:=false
ros2 run rqt_image_view rqt_image_view /detections/image_annotated
```

Manual teleop:
```bash
ros2 launch embodied_mvp mvp.launch.py enable_yolo:=false enable_search:=false
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Hardware bring-up checklist (Day 1)

Before flipping backends from `mock` to real, confirm:

| Item | What to find | Param to set |
|------|--------------|--------------|
| Motor protocol | UART text / I2C / PCA9685 + GPIO direction | `motor_node.backend`, `serial_port` |
| Wheelbase | measure half-distances (m) | `wheel_base_lx`, `wheel_base_ly` |
| IR sensor | digital binary / analog (ADC) / I2C / UART | `ir_node.backend` |
| Servo board | PCA9685 I2C address (0x40 default) | `pantilt_node.i2c_address` |
| Servo pulse range | check servo datasheet (us_min/us_max) | `pantilt_node.us_min/us_max` |

## NCNN speedup (when ultralytics too slow on Pi5)

On the laptop:
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=ncnn imgsz=320 int8=True
# scp yolov8n_ncnn_model/  pi@<pi-ip>:~/embodied_ws/
```

On Pi5, set in `params.yaml`:
```yaml
yolo_node:
  ros__parameters:
    backend: ncnn
    model_path: /home/pi/embodied_ws/yolov8n_ncnn_model
```

## Topic map

| Topic | Type | Direction |
|-------|------|-----------|
| `/camera/image_raw` | sensor_msgs/Image | camera → yolo |
| `/detections` | vision_msgs/Detection2DArray | yolo → search |
| `/detections/image_annotated` | sensor_msgs/Image | yolo → rviz2 |
| `/ir/range` | sensor_msgs/Range | ir → search |
| `/cmd_vel` | geometry_msgs/Twist | search → motor |
| `/pantilt/cmd` | geometry_msgs/Vector3 | search → pantilt |

## End-to-end verification (room test)

1. Place a chair 2-3 m from car
2. `ros2 launch embodied_mvp mvp.launch.py target_class:=chair`
3. Expect: rotate → spot chair → drive forward, recentering → IR < 0.4 m → stop, log `Found chair!`

Failure modes and tuning are listed in the plan file.

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

## TODO

- **Wire up side-IR obstacle avoidance.** The left/right binary IR sensors
  (GPIO 12/16) are read into `obs_left` / `obs_right` and published on
  `/ir/left` `/ir/right`, but no behavior currently uses them — the
  avoidance logic was dropped when APPROACHING was rewritten to discrete
  pulses. Today the robot only *appears* to avoid obstacles: the visual
  servo steers toward the target bbox and the path curves past obstacles
  incidentally. An obstacle that directly blocks the path or hides the
  target can stall or collide the robot. Add active avoidance in the
  SEARCH/APPROACHING tick: on a side-IR hit, steer away before the next
  forward pulse. Params `side_ir_enabled` and `avoid_yaw_bias` already exist.
