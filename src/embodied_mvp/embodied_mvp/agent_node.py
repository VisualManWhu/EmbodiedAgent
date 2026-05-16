"""Search behavior FSM. Hunts for `target_class` via visual servoing + IR stop.

Modes:
  IDLE          — robot stopped, waiting for a command.
  MANUAL        — phone-bot is driving; execute one timed movement step.
  SEARCH        — autonomous object search (sub-states SEARCHING/APPROACHING).
  ROTATE_PHOTO  — rotate 360° in four 90-deg steps, dwelling for a photo each step.

Search sub-states (active only while mode == SEARCH):
  SEARCHING   — no recent target: rotate in place, pan-tilt scans up/down.
  APPROACHING — target seen: P-controller drives bbox center to image center and bbox area to target_area.

Re-acquire window: target lost > lost_timeout_sec → back to SEARCHING.
Search timeout: every search_timeout_sec without a sighting → log a warning and
keep searching (non-terminal; only arrival at the target stops the robot).
"""
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')

        self.declare_parameter('target_class', 'chair')
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('control_rate_hz', 10.0)

        self.declare_parameter('kp_yaw', 0.004)
        self.declare_parameter('kp_fwd', 0.6)
        self.declare_parameter('max_fwd_speed', 0.18)
        self.declare_parameter('max_yaw_speed', 0.6)
        self.declare_parameter('yaw_deadband_px', 20.0)

        self.declare_parameter('search_yaw_speed', 0.4)
        self.declare_parameter('search_pitch_sweep_rad', 0.35)
        self.declare_parameter('search_pitch_period_sec', 4.0)
        # Stop-and-scan: rotate a short burst, then dwell still so the
        # low-fps detector gets sharp, motion-blur-free frames.
        self.declare_parameter('search_rotate_sec', 0.5)
        self.declare_parameter('search_dwell_sec', 1.3)
        # After losing a target mid-approach, sweep a local sector (oscillate)
        # instead of doing full circles — the target is usually still ahead.
        self.declare_parameter('reacquire_sec', 9.0)
        self.declare_parameter('reacquire_sweep_bursts', 6)

        # APPROACHING does one discrete pulse per fresh detection.
        self.declare_parameter('fresh_window_sec', 1.2)   # max detection age to act on
        self.declare_parameter('approach_step_sec', 0.3)     # rotation pulse duration
        self.declare_parameter('approach_fwd_step_sec', 0.7) # forward pulse duration (longer = faster)
        self.declare_parameter('approach_yaw_speed', 0.4)    # rotation pulse speed
        self.declare_parameter('approach_fwd_speed', 0.15)   # forward pulse speed
        self.declare_parameter('recenter_px', 160.0)         # |err_x| beyond this -> pure re-aim rotation
        self.declare_parameter('approach_steer_gain', 0.0012)# forward-pulse steering gain (rad/s per px)
        self.declare_parameter('approach_steer_max', 0.22)   # cap on forward-pulse steering
        self.declare_parameter('stop_distance_m', 0.4)
        self.declare_parameter('arrived_height_ratio', 0.55)  # bbox height / image height -> "close enough"
        self.declare_parameter('det_conf_min', 0.4)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('lost_timeout_sec', 1.0)
        self.declare_parameter('search_timeout_sec', 60.0)

        # Side IR obstacle avoidance (binary left/right sensors).
        self.declare_parameter('side_ir_enabled', True)
        self.declare_parameter('avoid_yaw_bias', 0.35)   # rad/s steer away per side hit

        # Mode FSM / command server params.
        self.declare_parameter('command_port', 9091)
        self.declare_parameter('manual_step_sec', 1.0)
        self.declare_parameter('manual_fwd_speed', 0.15)
        self.declare_parameter('manual_strafe_speed', 0.15)
        self.declare_parameter('manual_yaw_speed', 0.5)
        self.declare_parameter('rotate_90_sec', 1.4)
        self.declare_parameter('photo_dwell_sec', 3.0)
        self.declare_parameter('photo_settle_sec', 1.2)  # wait after a turn before the photo (sharp frame)

        p = self.get_parameter
        self.target_class = p('target_class').value
        self.W = int(p('image_width').value)
        self.H = int(p('image_height').value)
        self.cx_target = self.W / 2.0
        self.frame_area = float(self.W * self.H)

        self.Kp_yaw = p('kp_yaw').value
        self.Kp_fwd = p('kp_fwd').value
        self.v_max = p('max_fwd_speed').value
        self.w_max = p('max_yaw_speed').value
        self.yaw_db = p('yaw_deadband_px').value
        self.search_w = p('search_yaw_speed').value
        self.sweep = p('search_pitch_sweep_rad').value
        self.sweep_T = p('search_pitch_period_sec').value
        self.rotate_sec = p('search_rotate_sec').value
        self.dwell_sec = p('search_dwell_sec').value
        self.reacquire_sec = p('reacquire_sec').value
        self.reacquire_bursts = int(p('reacquire_sweep_bursts').value)
        self.fresh_window = p('fresh_window_sec').value
        self.step_sec = p('approach_step_sec').value
        self.fwd_step_sec = p('approach_fwd_step_sec').value
        self.approach_yaw = p('approach_yaw_speed').value
        self.approach_fwd = p('approach_fwd_speed').value
        self.recenter_px = p('recenter_px').value
        self.steer_gain = p('approach_steer_gain').value
        self.steer_max = p('approach_steer_max').value
        self.stop_d = p('stop_distance_m').value
        self.arrived_h_ratio = p('arrived_height_ratio').value
        self.conf_min = p('det_conf_min').value
        self.confirm_n = int(p('confirm_frames').value)
        self.lost_to = p('lost_timeout_sec').value
        self.search_to = p('search_timeout_sec').value
        self.side_ir_enabled = p('side_ir_enabled').value
        self.avoid_bias = p('avoid_yaw_bias').value

        self.manual_step_sec = p('manual_step_sec').value
        self.manual_fwd = p('manual_fwd_speed').value
        self.manual_strafe = p('manual_strafe_speed').value
        self.manual_yaw = p('manual_yaw_speed').value
        self.rotate_90_sec = p('rotate_90_sec').value
        self.photo_dwell_sec = p('photo_dwell_sec').value
        self.photo_settle_sec = p('photo_settle_sec').value

        self.state = 'SEARCHING'
        self.last_target = None
        self.last_target_time = 0.0
        self.start_time = time.time()
        self.consec_hits = 0
        self.ir_range = float('inf')
        self.obs_left = False
        self.obs_right = False
        # Stop-and-scan phase tracking.
        self.scan_phase = 'rotate'
        self.scan_phase_start = time.time()
        self.scan_dir = 1.0          # +1 / -1 rotation direction
        self.scan_bursts = 0         # rotate-bursts since last direction flip
        self.reacquire_until = 0.0   # local-sweep mode active until this time
        # APPROACHING discrete-pulse state.
        self.burst_end = 0.0
        self.burst_vx = 0.0
        self.burst_wz = 0.0
        self.last_acted_time = 0.0
        self._last_dbg = 0.0
        self._det_msgs = 0      # total /detections messages seen
        self._tgt_msgs = 0      # messages containing the target class

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pt_pub = self.create_publisher(Vector3, '/pantilt/cmd', 10)

        self.create_subscription(Detection2DArray, '/detections', self.on_det, 10)
        self.create_subscription(Range, '/ir/range', self.on_ir, 10)
        self.create_subscription(Image, '/camera/image_raw', self.on_image_meta, 1)
        self.create_subscription(Bool, '/ir/left', self.on_ir_left, 10)
        self.create_subscription(Bool, '/ir/right', self.on_ir_right, 10)

        period = 1.0 / float(p('control_rate_hz').value)
        self.create_timer(period, self.tick)
        self.get_logger().info(f'agent_node hunting target_class="{self.target_class}"')

        from embodied_mvp.command_server import CommandServer
        self.mode = 'IDLE'
        self.manual_twist = (0.0, 0.0, 0.0)   # vx, vy, wz
        self.manual_deadline = 0.0
        self.rp_index = 0
        self.rp_phase = 'dwell'
        self.rp_phase_start = 0.0
        self.rp_photo_posted = False
        self.cmd_server = CommandServer(self.get_parameter('command_port').value)
        self.get_logger().info(
            f"agent_node command server on :{self.get_parameter('command_port').value}")

    def on_image_meta(self, msg: Image):
        if msg.width != self.W or msg.height != self.H:
            self.W = msg.width
            self.H = msg.height
            self.cx_target = self.W / 2.0
            self.frame_area = float(self.W * self.H)

    def on_ir(self, msg: Range):
        self.ir_range = msg.range

    def on_ir_left(self, msg: Bool):
        self.obs_left = msg.data

    def on_ir_right(self, msg: Bool):
        self.obs_right = msg.data

    def on_det(self, msg: Detection2DArray):
        self._det_msgs += 1
        best = None
        best_score = 0.0
        for d in msg.detections:
            if not d.results:
                continue
            h = d.results[0]
            if h.hypothesis.class_id != self.target_class:
                continue
            if h.hypothesis.score < self.conf_min:
                continue
            if h.hypothesis.score > best_score:
                best_score = h.hypothesis.score
                best = d

        if best is not None:
            self._tgt_msgs += 1
            self.last_target = {
                'cx': best.bbox.center.position.x,
                'cy': best.bbox.center.position.y,
                'w': best.bbox.size_x,
                'h': best.bbox.size_y,
                'score': best_score,
            }
            self.last_target_time = time.time()
            # Build up, capped — tolerant of intermittent detection.
            self.consec_hits = min(self.confirm_n * 3, self.consec_hits + 1)
        else:
            # Decay (not hard reset) so a single missed frame doesn't
            # drop us out of lock during rotation / motion blur.
            self.consec_hits = max(0, self.consec_hits - 1)

    def publish_cmd(self, vx: float, wz: float):
        t = Twist()
        t.linear.x = max(-self.v_max, min(self.v_max, vx))
        t.angular.z = max(-self.w_max, min(self.w_max, wz))
        self.cmd_pub.publish(t)

    def publish_full_cmd(self, vx, vy, wz):
        t = Twist()
        t.linear.x = vx
        t.linear.y = vy
        t.angular.z = wz
        self.cmd_pub.publish(t)

    def publish_pantilt(self, yaw: float, pitch: float):
        v = Vector3()
        v.x = yaw
        v.y = pitch
        self.pt_pub.publish(v)

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
            self.rp_photo_posted = False
            # photo_ready:1 is posted by tick_rotate_photo after the settle delay
        self.get_logger().info(f'command {cmd} -> mode {self.mode}')

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
        self.burst_vx = 0.0
        self.burst_wz = 0.0
        self.last_acted_time = 0.0

    def tick_search(self):
        now = time.time()
        elapsed_since_seen = now - self.last_target_time if self.last_target else float('inf')

        # Throttled diagnostics — shows why a transition does/doesn't happen.
        if now - self._last_dbg >= 1.0:
            self._last_dbg = now
            self.get_logger().info(
                f'state={self.state} scan={self.scan_phase} '
                f'/detections msgs={self._det_msgs} with_target={self._tgt_msgs} '
                f'consec_hits={self.consec_hits} seen_ago={elapsed_since_seen:.1f}s '
                f'ir={self.ir_range:.2f}')

        # Time-windowed lock: detections arrive sparsely (target is a small
        # fraction of the high-rate /detections stream), so a per-message
        # consecutive-hit counter can never accumulate. Lock purely on
        # "target seen within lost_to seconds".
        has_recent_target = (
            self.last_target is not None
            and elapsed_since_seen < self.lost_to
        )

        if now - self.start_time > self.search_to and self.state == 'SEARCHING':
            # Non-terminal: log and keep searching (don't lock up the robot).
            self.get_logger().warn(
                f'Still hunting "{self.target_class}" — {self.search_to:.0f}s elapsed, continuing...')
            self.start_time = now

        if has_recent_target:
            self.state = 'APPROACHING'
        elif self.state == 'APPROACHING':
            # Lost the target mid-approach — re-scan locally first.
            self.state = 'SEARCHING'
            self.reacquire_until = now + self.reacquire_sec
            self.scan_bursts = 0

        if self.state == 'SEARCHING':
            # Stop-and-scan: alternate a rotate burst and a still dwell.
            # The dwell gives the low-fps detector blur-free frames.
            # Just after a loss (reacquire window) the rotation direction
            # flips every few bursts -> sweeps a local sector instead of
            # doing full circles, since the target is usually still ahead.
            reacquiring = now < self.reacquire_until
            wz = self.search_w * self.scan_dir
            phase_t = now - self.scan_phase_start
            if self.scan_phase == 'rotate':
                if phase_t >= self.rotate_sec:
                    self.scan_phase = 'dwell'
                    self.scan_phase_start = now
                    self.scan_bursts += 1
                    if reacquiring and self.scan_bursts >= self.reacquire_bursts:
                        self.scan_dir = -self.scan_dir   # flip sweep direction
                        self.scan_bursts = 0
                    self.publish_cmd(0.0, 0.0)
                else:
                    self.publish_cmd(0.0, wz)
            else:  # dwell
                if phase_t >= self.dwell_sec:
                    self.scan_phase = 'rotate'
                    self.scan_phase_start = now
                    self.publish_cmd(0.0, wz)
                else:
                    self.publish_cmd(0.0, 0.0)
            pitch = self.sweep * math.sin(2 * math.pi * now / self.sweep_T)
            self.publish_pantilt(0.0, pitch)
            return

        # APPROACHING — discrete corrective pulses.
        # Detection is sparse (~0.5 Hz) and latent (~1 s). A continuous servo
        # chases a stale, lagged bbox -> overshoot / hunting. Instead: do ONE
        # short fixed-size pulse per fresh detection, then fully stop and wait
        # for the next detection. Each pulse is small enough to keep the
        # target in frame; standing still between pulses keeps frames sharp.
        self.publish_pantilt(0.0, 0.0)

        # Arrival check (used mid-pulse AND between pulses). The ultrasonic
        # beam often misses a thin bottle, so distance is judged primarily by
        # how tall the target's bbox is in the frame (a reliable proxy that
        # does not depend on the sonar). IR is only a secondary trigger.
        height_frac = (self.last_target['h'] / self.H) if self.last_target else 0.0
        fresh = (self.last_target is not None
                 and elapsed_since_seen < self.fresh_window)
        arrived = (fresh and height_frac >= self.arrived_h_ratio) \
            or (self.ir_range < self.stop_d)
        if arrived:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().info(
                f'Found {self.target_class}! height_frac={height_frac:.2f} '
                f'ir={self.ir_range:.2f}m')
            self.cmd_server.post_event(f'arrived:{self.target_class}')
            self.mode = 'IDLE'
            self.state = 'SEARCHING'
            return

        if now < self.burst_end:
            # mid-pulse — keep executing the committed move
            self.publish_cmd(self.burst_vx, self.burst_wz)
            return

        # pulse finished — stop
        self.publish_cmd(0.0, 0.0)

        # start a new pulse only on a fresh detection we haven't acted on yet
        if (self.last_target is None
                or self.last_target_time <= self.last_acted_time
                or elapsed_since_seen > self.fresh_window):
            return

        t = self.last_target
        self.last_acted_time = self.last_target_time
        err_x = t['cx'] - self.cx_target

        if abs(err_x) > self.recenter_px:
            # target near the frame edge -> pure rotation pulse to re-aim
            self.burst_wz = -self.approach_yaw if err_x > 0 else self.approach_yaw
            self.burst_vx = 0.0
            self.burst_end = now + self.step_sec
        else:
            # target roughly ahead -> drive forward, gently steering toward
            # centre at the same time (no stop-and-rotate wobble)
            self.burst_vx = self.approach_fwd
            if abs(err_x) > self.yaw_db:
                steer = -self.steer_gain * err_x
                self.burst_wz = max(-self.steer_max, min(self.steer_max, steer))
            else:
                self.burst_wz = 0.0
            self.burst_end = now + self.fwd_step_sec

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
            self.publish_full_cmd(0.0, 0.0, 0.0)
            # Post the photo event only AFTER the car has settled, so the
            # camera frame is sharp (not blurred from the just-ended turn).
            if phase_t >= self.photo_settle_sec and not self.rp_photo_posted:
                self.cmd_server.post_event(f'photo_ready:{self.rp_index + 1}')
                self.rp_photo_posted = True
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
                    self.rp_photo_posted = False

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


def main():
    rclpy.init()
    node = AgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_cmd(0.0, 0.0)
        node.cmd_server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
