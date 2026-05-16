"""Forward distance sensor reader. Publishes sensor_msgs/Range on /ir/range.

Topic name kept as /ir/range (search_node subscribes it) — it just means
"forward proximity distance", regardless of sensor type.

Backends (param `backend`):
  - mock:       constant max range (no obstacle). Safe for desk testing.
  - ultrasonic: HC-SR04 via gpiozero DistanceSensor. Makerobo kit wiring:
                trigger=GPIO20, echo=GPIO21 (kit notebooks 5/6).

gpiozero DistanceSensor.distance returns metres (0 .. max_distance).
"""
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range


class IrNode(Node):
    def __init__(self):
        super().__init__('ir_node')

        self.declare_parameter('backend', 'mock')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('min_range_m', 0.02)
        self.declare_parameter('max_range_m', 3.0)
        self.declare_parameter('field_of_view_rad', 0.26)
        self.declare_parameter('frame_id', 'ultrasonic_link')
        self.declare_parameter('smoothing_window', 5)
        self.declare_parameter('trigger_pin', 20)
        self.declare_parameter('echo_pin', 21)

        self.backend = self.get_parameter('backend').value
        self.min_r = self.get_parameter('min_range_m').value
        self.max_r = self.get_parameter('max_range_m').value
        self.fov = self.get_parameter('field_of_view_rad').value
        self.frame_id = self.get_parameter('frame_id').value
        self.win = max(1, int(self.get_parameter('smoothing_window').value))
        self._buf = []

        self._sensor = None
        self._init_backend()

        self.pub = self.create_publisher(Range, '/ir/range', 10)
        period = 1.0 / self.get_parameter('rate_hz').value
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f'ir_node ready, backend={self.backend}')

    def _init_backend(self):
        if self.backend == 'mock':
            return
        if self.backend == 'ultrasonic':
            # Pi5 needs the lgpio pin factory for gpiozero.
            os.environ.setdefault('GPIOZERO_PIN_FACTORY', 'lgpio')
            from gpiozero import DistanceSensor
            self._sensor = DistanceSensor(
                echo=self.get_parameter('echo_pin').value,
                trigger=self.get_parameter('trigger_pin').value,
                max_distance=self.max_r,
            )
        else:
            raise RuntimeError(f'unknown backend: {self.backend}')

    def read_raw(self) -> float:
        if self.backend == 'mock':
            # Constant "clear" — a sine wave would dip below stop_distance
            # and trigger false arrivals in search_node.
            return self.max_r
        if self.backend == 'ultrasonic' and self._sensor is not None:
            return float(self._sensor.distance)  # metres
        return self.max_r

    def smooth(self, x: float) -> float:
        self._buf.append(x)
        if len(self._buf) > self.win:
            self._buf.pop(0)
        return sum(self._buf) / len(self._buf)

    def tick(self):
        raw = self.read_raw()
        d = max(self.min_r, min(self.max_r, raw))
        d = self.smooth(d)
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = self.fov
        msg.min_range = self.min_r
        msg.max_range = self.max_r
        msg.range = float(d)
        self.pub.publish(msg)

    def destroy_node(self):
        if self._sensor is not None:
            try:
                self._sensor.close()
            except Exception:
                pass
        super().destroy_node()


def main():
    rclpy.init()
    node = IrNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
