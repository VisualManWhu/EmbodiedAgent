"""Detection bridge — Pi side of offloaded detection.

The laptop GPU detector POSTs detection results here over HTTP; this node
republishes them as /detections so search_node works unchanged. Running YOLO
on the laptop instead of the Pi keeps Pi5 CPU (and thus current draw) low,
which avoids the motor-battery BMS overcurrent trip.

POST http://<pi>:9090/detections  with JSON body:
  {"width": 640, "height": 480,
   "detections": [
       {"cls": "bottle", "score": 0.81, "cx": 320, "cy": 240, "w": 90, "h": 200},
       ...
   ]}
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from vision_msgs.msg import (Detection2DArray, Detection2D,
                             ObjectHypothesisWithPose, BoundingBox2D)


class DetBridgeNode(Node):
    def __init__(self):
        super().__init__('det_bridge_node')

        self.declare_parameter('port', 9090)
        self.port = int(self.get_parameter('port').value)

        self.pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.lock = threading.Lock()
        self.pending = None

        self._start_http()
        # Publish on the ROS executor thread (one publish per received POST).
        self.create_timer(0.02, self._flush)
        self.get_logger().info(
            f'det_bridge_node listening on :{self.port}/detections')

    def _build(self, payload) -> Detection2DArray:
        arr = Detection2DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'camera_link'
        for d in payload.get('detections', []):
            det = Detection2D()
            det.header = arr.header
            bb = BoundingBox2D()
            bb.center.position.x = float(d['cx'])
            bb.center.position.y = float(d['cy'])
            bb.size_x = float(d['w'])
            bb.size_y = float(d['h'])
            det.bbox = bb
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(d['cls'])
            hyp.hypothesis.score = float(d['score'])
            det.results.append(hyp)
            arr.detections.append(det)
        return arr

    def _flush(self):
        with self.lock:
            arr = self.pending
            self.pending = None
        if arr is not None:
            self.pub.publish(arr)

    def _start_http(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    payload = json.loads(self.rfile.read(n))
                    arr = node._build(payload)
                    with node.lock:
                        node.pending = arr
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                except Exception as e:  # noqa: BLE001
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(e).encode())

        self._server = ThreadingHTTPServer(('0.0.0.0', self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def destroy_node(self):
        try:
            self._server.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DetBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
