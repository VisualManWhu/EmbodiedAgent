"""Minimal MJPEG HTTP streamer. Subscribes a ROS Image topic, serves it as
multipart MJPEG so any browser (no ROS client needed) can view it live.

View on laptop:  http://<pi-ip>:8080/

Params:
  topic   : image topic to stream (default /detections/image_annotated)
  port    : HTTP port (default 8080)
  quality : JPEG quality 1-100 (default 60)
  fps_cap : max frames pushed per client per second (default 20)
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MjpegNode(Node):
    def __init__(self):
        super().__init__('mjpeg_node')

        self.declare_parameter('topic', '/detections/image_annotated')
        self.declare_parameter('port', 8080)
        self.declare_parameter('quality', 60)
        self.declare_parameter('fps_cap', 20.0)

        self.topic = self.get_parameter('topic').value
        self.port = int(self.get_parameter('port').value)
        self.quality = int(self.get_parameter('quality').value)
        self.frame_dt = 1.0 / float(self.get_parameter('fps_cap').value)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = None  # latest JPEG bytes

        self.create_subscription(Image, self.topic, self.on_image, 5)
        self._start_http()
        self.get_logger().info(
            f'mjpeg_node streaming "{self.topic}" at http://0.0.0.0:{self.port}/')

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return
        ok, jpg = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.lock:
                self.latest = jpg.tobytes()

    def _start_http(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    'Content-Type',
                    'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                try:
                    while rclpy.ok():
                        with node.lock:
                            jpg = node.latest
                        if jpg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(
                            f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b'\r\n')
                        time.sleep(node.frame_dt)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(('0.0.0.0', self.port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

    def destroy_node(self):
        try:
            self._server.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = MjpegNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
