"""YOLOv8 detector node. Subscribes /camera/image_raw, publishes /detections + annotated image.

Backends (param `backend`):
  - ultralytics: PyTorch via `ultralytics` (default; easy, slower on Pi5).
  - ncnn:        NCNN INT8 model dir (faster on Pi5, ~8-10 fps @ 320).

Export NCNN on laptop:
    yolo export model=yolov8n.pt format=ncnn imgsz=320 int8=True
    → yolov8n_ncnn_model/ (copy to Pi5)
"""
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('backend', 'ultralytics')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('classes_whitelist', [''])
        self.declare_parameter('inference_every_n', 1)

        self.imgsz = self.get_parameter('imgsz').value
        self.conf = self.get_parameter('conf_threshold').value
        self.iou = self.get_parameter('iou_threshold').value
        self.publish_annotated = self.get_parameter('publish_annotated').value
        wl = [c for c in self.get_parameter('classes_whitelist').value if c]
        self.whitelist = set(wl) if wl else None
        self.every_n = max(1, int(self.get_parameter('inference_every_n').value))
        self._frame_idx = 0

        self.bridge = CvBridge()
        self._init_model()

        self.det_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        if self.publish_annotated:
            self.img_pub = self.create_publisher(Image, '/detections/image_annotated', 5)

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.on_image, 5)

        self._last_log = time.time()
        self._frame_count = 0
        self.get_logger().info(f'yolo_node ready, backend={self.get_parameter("backend").value}, imgsz={self.imgsz}')

    def _init_model(self):
        backend = self.get_parameter('backend').value
        path = self.get_parameter('model_path').value
        from ultralytics import YOLO
        # ultralytics auto-loads NCNN if path is a *_ncnn_model dir
        self.model = YOLO(path)
        self.names = self.model.names

    def on_image(self, msg: Image):
        self._frame_idx += 1
        if self._frame_idx % self.every_n != 0:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        results = self.model.predict(
            source=frame, imgsz=self.imgsz,
            conf=self.conf, iou=self.iou, verbose=False,
        )[0]

        out = Detection2DArray()
        out.header = msg.header

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                name = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
                if self.whitelist and name not in self.whitelist:
                    continue
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)

                d = Detection2D()
                d.header = msg.header
                bb = BoundingBox2D()
                bb.center.position.x = cx
                bb.center.position.y = cy
                bb.size_x = w
                bb.size_y = h
                d.bbox = bb

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = name
                hyp.hypothesis.score = conf
                d.results.append(hyp)
                out.detections.append(d)

        self.det_pub.publish(out)

        if self.publish_annotated:
            try:
                annotated = results.plot()
                img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                img_msg.header = msg.header
                self.img_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().warn(f'annotate failed: {e}')

        self._frame_count += 1
        now = time.time()
        if now - self._last_log >= 5.0:
            fps = self._frame_count / (now - self._last_log)
            self.get_logger().info(f'inference fps={fps:.1f}')
            self._frame_count = 0
            self._last_log = now


def main():
    rclpy.init()
    node = YoloNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
