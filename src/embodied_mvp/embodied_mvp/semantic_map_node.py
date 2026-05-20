"""Semantic map bridge — Pi side of the semantic-SLAM viewer.

The laptop builds the semantic map (AprilTag localization + multi-view-fused
object landmarks) and POSTs it here as JSON; this node republishes it as RViz
markers and broadcasts the robot TF, so the map is viewable in RViz2 without
the laptop needing a ROS install.

POST http://<pi>:9092/map  with JSON body:
  {"robot": {"x": 1.2, "y": 0.4, "yaw": 0.3},          # may be null
   "tags": [{"id": 0, "x": 0.0, "y": 0.0, "z": 0.5}, ...],
   "landmarks": [{"id": 1, "label": "chair", "x": 2.1, "y": 0.6, "z": 0.0,
                  "state": "CONFIRMED", "confidence": 0.9}, ...]}

Markers distinguish CONFIRMED (solid) from TENTATIVE (translucent) landmarks.
"""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__('semantic_map_node')

        self.declare_parameter('port', 9092)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('camera_height_m', 0.30)
        self.port = int(self.get_parameter('port').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.cam_height = float(self.get_parameter('camera_height_m').value)

        self.pub = self.create_publisher(MarkerArray, '/semantic_map/markers', 1)
        self.tf = TransformBroadcaster(self)

        self.lock = threading.Lock()
        self.payload = {'robot': None, 'tags': [], 'landmarks': []}

        self._start_http()
        self.create_timer(0.2, self._publish)   # 5 Hz republish
        self.get_logger().info(
            f'semantic_map_node listening on :{self.port}/map')

    # ---- marker construction ---------------------------------------------

    def _publish(self):
        with self.lock:
            data = self.payload

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        stamp = self.get_clock().now().to_msg()
        mid = 0

        for tag in data.get('tags', []):
            m = self._marker(mid, 'tags', stamp)
            m.type = Marker.CUBE
            m.pose.position = Point(x=float(tag['x']), y=float(tag['y']),
                                    z=float(tag.get('z', 0.0)))
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.16, 0.16, 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
            arr.markers.append(m)
            mid += 1

        for lm in data.get('landmarks', []):
            confirmed = lm.get('state') == 'CONFIRMED'
            m = self._marker(mid, 'landmarks', stamp)
            m.type = Marker.SPHERE
            m.pose.position = Point(x=float(lm['x']), y=float(lm['y']),
                                    z=float(lm.get('z', 0.0)) + 0.1)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            if confirmed:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.8, 0.1, 0.4
            arr.markers.append(m)
            mid += 1

            txt = self._marker(mid, 'labels', stamp)
            txt.type = Marker.TEXT_VIEW_FACING
            txt.pose.position = Point(x=float(lm['x']), y=float(lm['y']),
                                      z=float(lm.get('z', 0.0)) + 0.35)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.18
            txt.color.r = txt.color.g = txt.color.b = 1.0
            txt.color.a = 1.0 if confirmed else 0.5
            txt.text = f"{lm.get('label', '?')} ({lm.get('confidence', 0):.0%})"
            arr.markers.append(txt)
            mid += 1

        robot = data.get('robot')
        if robot is not None:
            m = self._marker(mid, 'robot', stamp)
            m.type = Marker.ARROW
            m.pose.position = Point(x=float(robot['x']), y=float(robot['y']),
                                    z=0.05)
            yaw = float(robot.get('yaw', 0.0))
            m.pose.orientation.z = math.sin(yaw / 2.0)
            m.pose.orientation.w = math.cos(yaw / 2.0)
            m.scale.x, m.scale.y, m.scale.z = 0.3, 0.08, 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 1.0
            arr.markers.append(m)
            self._broadcast_tf(robot, stamp)

        self.pub.publish(arr)

    def _marker(self, mid, ns, stamp):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.action = Marker.ADD
        return m

    def _broadcast_tf(self, robot, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(robot['x'])
        t.transform.translation.y = float(robot['y'])
        t.transform.translation.z = 0.0
        yaw = float(robot.get('yaw', 0.0))
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf.sendTransform(t)

        c = TransformStamped()
        c.header.stamp = stamp
        c.header.frame_id = 'base_link'
        c.child_frame_id = 'camera'
        c.transform.translation.z = self.cam_height
        c.transform.rotation.w = 1.0
        self.tf.sendTransform(c)

    # ---- HTTP ------------------------------------------------------------

    def _start_http(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not self.path.startswith('/map'):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    payload = json.loads(self.rfile.read(n))
                    with node.lock:
                        node.payload = payload
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
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
