"""Side IR obstacle sensors (binary). Publishes /ir/left and /ir/right (std_msgs/Bool).

Makerobo kit (notebooks 4/6): left IR = GPIO12, right IR = GPIO16, pull_up=True.
gpiozero Button .value == 1 means obstacle detected (kit convention).

Backends (param `backend`):
  - mock: always False (clear).
  - gpio: real gpiozero Button reads.
"""
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class SideIrNode(Node):
    def __init__(self):
        super().__init__('side_ir_node')

        self.declare_parameter('backend', 'mock')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('left_pin', 12)
        self.declare_parameter('right_pin', 16)

        self.backend = self.get_parameter('backend').value

        self._left = None
        self._right = None
        self._init_backend()

        self.pub_l = self.create_publisher(Bool, '/ir/left', 10)
        self.pub_r = self.create_publisher(Bool, '/ir/right', 10)
        period = 1.0 / self.get_parameter('rate_hz').value
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f'side_ir_node ready, backend={self.backend}')

    def _init_backend(self):
        if self.backend == 'mock':
            return
        if self.backend == 'gpio':
            os.environ.setdefault('GPIOZERO_PIN_FACTORY', 'lgpio')
            from gpiozero import Button
            self._left = Button(self.get_parameter('left_pin').value, pull_up=True)
            self._right = Button(self.get_parameter('right_pin').value, pull_up=True)
        else:
            raise RuntimeError(f'unknown backend: {self.backend}')

    def _obstacle(self, btn) -> bool:
        if btn is None:
            return False
        return btn.value == 1   # kit convention: 1 = obstacle

    def tick(self):
        self.pub_l.publish(Bool(data=self._obstacle(self._left)))
        self.pub_r.publish(Bool(data=self._obstacle(self._right)))

    def destroy_node(self):
        for s in (self._left, self._right):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        super().destroy_node()


def main():
    rclpy.init()
    node = SideIrNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
