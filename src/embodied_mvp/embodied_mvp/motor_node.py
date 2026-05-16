"""Mecanum motor driver. Subscribes /cmd_vel, writes wheel speeds via pluggable backend.

Backends (param `backend`):
  - mock:      log only, no hardware write. Safe for desk testing.
  - serial:    send "M:fl,fr,rl,rr\n" over UART. Set `serial_port`, `serial_baud`.
  - loborobot: Makerobo kit — PCA9685 @ 0x40 + GPIO 24/25 via vendored LOBOROBOT lib.

Kinematics (mecanum, +x forward, +y left, +z yaw CCW):
  v_fl = vx - vy - (Lx+Ly)*wz
  v_fr = vx + vy + (Lx+Ly)*wz
  v_rl = vx + vy - (Lx+Ly)*wz
  v_rr = vx - vy + (Lx+Ly)*wz

Slew-rate limiting: a control loop ramps wheel outputs toward the commanded
target at <= slew_per_sec. This kills the current inrush from a 0->full step,
which otherwise spikes battery draw and can trip the BMS / brown out the Pi.
"""
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')

        self.declare_parameter('backend', 'mock')
        self.declare_parameter('wheel_base_lx', 0.08)
        self.declare_parameter('wheel_base_ly', 0.085)
        self.declare_parameter('max_wheel_speed', 1.0)
        self.declare_parameter('serial_port', '/dev/ttyAMA0')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('cmd_timeout_sec', 0.5)
        # loborobot backend: normalized wheel speed -> PCA9685 duty percent.
        self.declare_parameter('speed_gain', 300.0)        # wheel_norm * gain = percent
        self.declare_parameter('min_move_percent', 35.0)   # floor to overcome motor stall
        self.declare_parameter('max_move_percent', 90.0)
        self.declare_parameter('move_eps', 0.01)           # below this -> treat as stop
        # slew-rate limiting (anti current-spike).
        self.declare_parameter('slew_per_sec', 2.5)        # max wheel_norm change / second
        self.declare_parameter('update_rate_hz', 50.0)

        self.backend = self.get_parameter('backend').value
        self.Lx = self.get_parameter('wheel_base_lx').value
        self.Ly = self.get_parameter('wheel_base_ly').value
        self.v_max = self.get_parameter('max_wheel_speed').value
        self.cmd_timeout = self.get_parameter('cmd_timeout_sec').value
        self.speed_gain = self.get_parameter('speed_gain').value
        self.min_pct = self.get_parameter('min_move_percent').value
        self.max_pct = self.get_parameter('max_move_percent').value
        self.move_eps = self.get_parameter('move_eps').value
        self.slew = self.get_parameter('slew_per_sec').value
        update_hz = self.get_parameter('update_rate_hz').value
        self.max_step = self.slew / update_hz

        self._init_backend()

        # [fl, fr, rl, rr]
        self.target = [0.0, 0.0, 0.0, 0.0]
        self.current = [0.0, 0.0, 0.0, 0.0]

        self.last_cmd_time = self.get_clock().now()
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.update_timer = self.create_timer(1.0 / update_hz, self.update)

        self.get_logger().info(f'motor_node ready, backend={self.backend}')

    def _init_backend(self):
        self._serial = None
        self._robot = None
        if self.backend == 'serial':
            import serial
            self._serial = serial.Serial(
                self.get_parameter('serial_port').value,
                self.get_parameter('serial_baud').value,
                timeout=0.1,
            )
        elif self.backend == 'loborobot':
            # Pi5 needs the lgpio pin factory for gpiozero (RPi.GPIO unsupported).
            os.environ.setdefault('GPIOZERO_PIN_FACTORY', 'lgpio')
            from embodied_mvp.loborobot_lib import LOBOROBOT
            self._robot = LOBOROBOT()
        elif self.backend == 'mock':
            pass
        else:
            raise RuntimeError(f'unknown backend: {self.backend}')

    # Mecanum wheel -> Makerobo motor index: FL=0, FR=1, RL=2, RR=3.
    def _drive(self, motor_idx, wheel_norm):
        if abs(wheel_norm) < self.move_eps:
            self._robot.MotorStop(motor_idx)
            return
        pct = max(self.min_pct, min(self.max_pct, abs(wheel_norm) * self.speed_gain))
        direction = 'forward' if wheel_norm > 0 else 'backward'
        self._robot.MotorRun(motor_idx, direction, pct)

    def mecanum(self, vx, vy, wz):
        L = self.Lx + self.Ly
        fl = vx - vy - L * wz
        fr = vx + vy + L * wz
        rl = vx + vy - L * wz
        rr = vx - vy + L * wz
        peak = max(abs(fl), abs(fr), abs(rl), abs(rr), 1e-6)
        if peak > self.v_max:
            s = self.v_max / peak
            fl, fr, rl, rr = fl*s, fr*s, rl*s, rr*s
        return [fl, fr, rl, rr]

    def on_cmd(self, msg: Twist):
        self.last_cmd_time = self.get_clock().now()
        self.target = self.mecanum(msg.linear.x, msg.linear.y, msg.angular.z)

    def update(self):
        # Watchdog: stale command -> coast target to zero (still ramped).
        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        if elapsed > self.cmd_timeout:
            self.target = [0.0, 0.0, 0.0, 0.0]

        # Slew current toward target, capped per step.
        for i in range(4):
            delta = self.target[i] - self.current[i]
            if delta > self.max_step:
                delta = self.max_step
            elif delta < -self.max_step:
                delta = -self.max_step
            self.current[i] += delta

        self.write_wheels(*self.current)

    def write_wheels(self, fl, fr, rl, rr):
        if self.backend == 'mock':
            self.get_logger().debug(f'wheels fl={fl:.2f} fr={fr:.2f} rl={rl:.2f} rr={rr:.2f}')
        elif self.backend == 'serial':
            line = f'M:{fl:.3f},{fr:.3f},{rl:.3f},{rr:.3f}\n'
            self._serial.write(line.encode())
        elif self.backend == 'loborobot':
            self._drive(0, fl)   # front-left
            self._drive(1, fr)   # front-right
            self._drive(2, rl)   # rear-left
            self._drive(3, rr)   # rear-right


def main():
    rclpy.init()
    node = MotorNode()
    try:
        rclpy.spin(node)
    finally:
        node.write_wheels(0.0, 0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
