"""Search behavior FSM. Hunts for `target_class` via visual servoing + IR stop.

States:
  SEARCHING   — no recent target: rotate in place, pan-tilt scans up/down.
  APPROACHING — target seen: P-controller drives bbox center to image center and bbox area to target_area.
  ARRIVED     — IR < stop_distance AND bbox area > target_area * arrived_ratio: zero velocity, announce.

Re-acquire window: target lost > lost_timeout_sec → back to SEARCHING.
Search timeout: every search_timeout_sec without a sighting → log a warning and
keep searching (non-terminal; only ARRIVED on the target stops the robot).
"""
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray


class SearchNode(Node):
    def __init__(self):
        super().__init__('search_node')

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

        self.declare_parameter('stop_distance_m', 0.4)
        self.declare_parameter('arrived_bbox_area_ratio', 0.25)
        self.declare_parameter('det_conf_min', 0.4)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('lost_timeout_sec', 1.0)
        self.declare_parameter('search_timeout_sec', 60.0)

        # Side IR obstacle avoidance (binary left/right sensors).
        self.declare_parameter('side_ir_enabled', True)
        self.declare_parameter('avoid_yaw_bias', 0.35)   # rad/s steer away per side hit

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
        self.stop_d = p('stop_distance_m').value
        self.arrived_ratio = p('arrived_bbox_area_ratio').value
        self.conf_min = p('det_conf_min').value
        self.confirm_n = int(p('confirm_frames').value)
        self.lost_to = p('lost_timeout_sec').value
        self.search_to = p('search_timeout_sec').value
        self.side_ir_enabled = p('side_ir_enabled').value
        self.avoid_bias = p('avoid_yaw_bias').value

        self.state = 'SEARCHING'
        self.last_target = None
        self.last_target_time = 0.0
        self.start_time = time.time()
        self.consec_hits = 0
        self.ir_range = float('inf')
        self.obs_left = False
        self.obs_right = False

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pt_pub = self.create_publisher(Vector3, '/pantilt/cmd', 10)

        self.create_subscription(Detection2DArray, '/detections', self.on_det, 10)
        self.create_subscription(Range, '/ir/range', self.on_ir, 10)
        self.create_subscription(Image, '/camera/image_raw', self.on_image_meta, 1)
        self.create_subscription(Bool, '/ir/left', self.on_ir_left, 10)
        self.create_subscription(Bool, '/ir/right', self.on_ir_right, 10)

        period = 1.0 / float(p('control_rate_hz').value)
        self.create_timer(period, self.tick)
        self.get_logger().info(f'search_node hunting target_class="{self.target_class}"')

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
            self.last_target = {
                'cx': best.bbox.center.position.x,
                'cy': best.bbox.center.position.y,
                'w': best.bbox.size_x,
                'h': best.bbox.size_y,
                'score': best_score,
            }
            self.last_target_time = time.time()
            self.consec_hits += 1
        else:
            self.consec_hits = 0

    def publish_cmd(self, vx: float, wz: float):
        t = Twist()
        t.linear.x = max(-self.v_max, min(self.v_max, vx))
        t.angular.z = max(-self.w_max, min(self.w_max, wz))
        self.cmd_pub.publish(t)

    def publish_pantilt(self, yaw: float, pitch: float):
        v = Vector3()
        v.x = yaw
        v.y = pitch
        self.pt_pub.publish(v)

    def tick(self):
        now = time.time()
        elapsed_since_seen = now - self.last_target_time if self.last_target else float('inf')

        has_recent_target = (
            self.last_target is not None
            and elapsed_since_seen < self.lost_to
            and self.consec_hits >= self.confirm_n
        )

        if self.state == 'ARRIVED':
            self.publish_cmd(0.0, 0.0)
            return

        if now - self.start_time > self.search_to and self.state == 'SEARCHING':
            # Non-terminal: log and keep searching (don't lock up the robot).
            self.get_logger().warn(
                f'Still hunting "{self.target_class}" — {self.search_to:.0f}s elapsed, continuing...')
            self.start_time = now

        if has_recent_target:
            self.state = 'APPROACHING'
        elif self.state == 'APPROACHING':
            self.state = 'SEARCHING'

        if self.state == 'SEARCHING':
            self.publish_cmd(0.0, self.search_w)
            pitch = self.sweep * math.sin(2 * math.pi * now / self.sweep_T)
            self.publish_pantilt(0.0, pitch)
            return

        # APPROACHING
        t = self.last_target
        err_x = t['cx'] - self.cx_target
        if abs(err_x) < self.yaw_db:
            wz = 0.0
        else:
            wz = -self.Kp_yaw * err_x

        area = t['w'] * t['h']
        area_frac = area / self.frame_area
        fwd_err = max(0.0, self.arrived_ratio - area_frac)
        vx = self.Kp_fwd * fwd_err

        if self.ir_range < self.stop_d:
            vx = 0.0
            if area_frac >= self.arrived_ratio * 0.7:
                self.publish_cmd(0.0, 0.0)
                self.get_logger().info(
                    f'Found {self.target_class}! ir={self.ir_range:.2f}m, area_frac={area_frac:.2f}, score={t["score"]:.2f}'
                )
                self.state = 'ARRIVED'
                return

        # Side IR avoidance: steer away from a blocked side while moving forward.
        if self.side_ir_enabled and vx > 0.0:
            if self.obs_left and not self.obs_right:
                wz -= self.avoid_bias      # obstacle left -> turn right
            elif self.obs_right and not self.obs_left:
                wz += self.avoid_bias      # obstacle right -> turn left
            elif self.obs_left and self.obs_right:
                vx = 0.0                   # both blocked -> stop forward, keep steering

        # face target with pan-tilt for visual feedback (optional centering aid)
        self.publish_pantilt(0.0, 0.0)
        self.publish_cmd(vx, wz)


def main():
    rclpy.init()
    node = SearchNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
