"""2-DOF pan-tilt servo driver. Subscribes /pantilt/cmd (Vector3: x=yaw_rad, y=pitch_rad).

Backends (param `backend`):
  - mock:      log only.
  - loborobot: Makerobo kit. Drives PCA9685 @ 0x40 servo channels directly
               (pan=ch10, tilt=ch9). Uses ONLY PCA9685 — not the full LOBOROBOT
               class — so it never claims GPIO 24/25 (motor_node owns those).

Kit servo convention (from notebook 11): angles in DEGREES.
  pan  center 90 deg,  range -90..180
  tilt center -10 deg, range -10..90
Angle->PWM count: 4096 * ((deg*11) + 500) / 20000   (same as LOBOROBOT.set_servo_angle)
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


def angle_to_count(deg: float) -> int:
    return int(4096 * ((deg * 11) + 500) / 20000)


class PanTiltNode(Node):
    def __init__(self):
        super().__init__('pantilt_node')

        self.declare_parameter('backend', 'mock')
        self.declare_parameter('i2c_address', 0x40)
        self.declare_parameter('pan_channel', 10)
        self.declare_parameter('tilt_channel', 9)
        self.declare_parameter('pan_center_deg', 90.0)
        self.declare_parameter('tilt_center_deg', -10.0)
        self.declare_parameter('pan_min_deg', -90.0)
        self.declare_parameter('pan_max_deg', 180.0)
        self.declare_parameter('tilt_min_deg', -10.0)
        self.declare_parameter('tilt_max_deg', 90.0)
        self.declare_parameter('pan_invert', False)
        self.declare_parameter('tilt_invert', False)
        self.declare_parameter('pwm_freq_hz', 50)

        p = self.get_parameter
        self.backend = p('backend').value
        self.pan_ch = p('pan_channel').value
        self.tilt_ch = p('tilt_channel').value
        self.pan_c = p('pan_center_deg').value
        self.tilt_c = p('tilt_center_deg').value
        self.pan_lo = p('pan_min_deg').value
        self.pan_hi = p('pan_max_deg').value
        self.tilt_lo = p('tilt_min_deg').value
        self.tilt_hi = p('tilt_max_deg').value
        self.pan_inv = p('pan_invert').value
        self.tilt_inv = p('tilt_invert').value

        self._pca = None
        self._init_backend()

        self.sub = self.create_subscription(Vector3, '/pantilt/cmd', self.on_cmd, 10)
        self.write(0.0, 0.0)  # center on start
        self.get_logger().info(f'pantilt_node ready, backend={self.backend}')

    def _init_backend(self):
        if self.backend == 'loborobot':
            from embodied_mvp.loborobot_lib import PCA9685
            self._pca = PCA9685(self.get_parameter('i2c_address').value)
            self._pca.setPWMFreq(self.get_parameter('pwm_freq_hz').value)
        elif self.backend != 'mock':
            raise RuntimeError(f'unknown backend: {self.backend}')

    def write(self, yaw_rad: float, pitch_rad: float):
        yaw = -yaw_rad if self.pan_inv else yaw_rad
        pitch = -pitch_rad if self.tilt_inv else pitch_rad
        pan_deg = self.pan_c + math.degrees(yaw)
        tilt_deg = self.tilt_c + math.degrees(pitch)
        pan_deg = max(self.pan_lo, min(self.pan_hi, pan_deg))
        tilt_deg = max(self.tilt_lo, min(self.tilt_hi, tilt_deg))

        if self.backend == 'mock':
            self.get_logger().debug(f'pan={pan_deg:.1f} tilt={tilt_deg:.1f} deg')
            return
        if self.backend == 'loborobot' and self._pca is not None:
            self._pca.setPWM(self.pan_ch, 0, angle_to_count(pan_deg))
            self._pca.setPWM(self.tilt_ch, 0, angle_to_count(tilt_deg))

    def on_cmd(self, msg: Vector3):
        self.write(msg.x, msg.y)

    def destroy_node(self):
        # Stop PWM on servo channels so the servo de-energizes (no buzz) on exit.
        if self.backend == 'loborobot' and self._pca is not None:
            try:
                self._pca.setPWM(self.pan_ch, 0, 0)
                self._pca.setPWM(self.tilt_ch, 0, 0)
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = PanTiltNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
