"""CSI camera node for Raspberry Pi 5.

RoboStack's conda libcamera can't enumerate the Pi5 CSI camera (missing PiSP
IPA modules). Workaround: spawn the SYSTEM `rpicam-vid` binary (which uses the
working system libcamera), read its MJPEG stdout stream, decode each JPEG with
OpenCV inside the conda env, and publish sensor_msgs/Image.

MJPEG (not raw YUV420) is used because cv2.imdecode produces unambiguous BGR —
raw YUV plane-order (I420 vs YV12) guesswork caused a colour cast.

No python/libcamera bindings are imported here — only a subprocess pipe — so
there is no ABI clash with the conda numpy/opencv.

Params:
  rpicam_bin   : binary name (rpicam-vid, fallback libcamera-vid)
  width/height : capture resolution
  framerate    : publish rate
  hflip/vflip  : image flips
  shutter_us   : fixed exposure in microseconds, 0 = auto
  gain         : fixed analogue gain, 0 = auto
  mjpeg_quality: rpicam-vid MJPEG quality 1-100
  frame_id     : header frame_id
"""
import os
import subprocess
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

SOI = b'\xff\xd8'   # JPEG start-of-image
EOI = b'\xff\xd9'   # JPEG end-of-image


class CsiCameraNode(Node):
    def __init__(self):
        super().__init__('camera')

        self.declare_parameter('rpicam_bin', 'rpicam-vid')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('framerate', 15)
        self.declare_parameter('hflip', False)
        self.declare_parameter('vflip', False)
        self.declare_parameter('shutter_us', 0)      # 0 = auto exposure
        self.declare_parameter('gain', 0.0)          # 0 = auto gain
        self.declare_parameter('mjpeg_quality', 80)
        self.declare_parameter('frame_id', 'camera_link')

        self.w = int(self.get_parameter('width').value)
        self.h = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('framerate').value)
        self.frame_id = self.get_parameter('frame_id').value

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 5)

        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._dead = False

        self._proc = self._start_rpicam()
        threading.Thread(target=self._reader_loop, daemon=True).start()
        self.create_timer(1.0 / max(1, self.fps), self.tick)
        self.get_logger().info(
            f'camera (CSI via rpicam-vid MJPEG) {self.w}x{self.h}@{self.fps} ready')

    def _system_env(self):
        """Env for the rpicam-vid subprocess: strip conda paths so the SYSTEM
        binary loads SYSTEM libs (conda libstdc++/libcamera would crash it)."""
        env = os.environ.copy()
        conda = env.get('CONDA_PREFIX', '')
        ld = env.get('LD_LIBRARY_PATH', '')
        if ld:
            kept = [p for p in ld.split(':') if p and (not conda or conda not in p)]
            if kept:
                env['LD_LIBRARY_PATH'] = ':'.join(kept)
            else:
                env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
        env['PATH'] = '/usr/bin:/bin:/usr/local/bin:' + env.get('PATH', '')
        return env

    def _start_rpicam(self):
        binname = self.get_parameter('rpicam_bin').value
        cmd = [
            binname, '-t', '0', '--nopreview',
            '--width', str(self.w), '--height', str(self.h),
            '--framerate', str(self.fps),
            '--codec', 'mjpeg',
            '--quality', str(int(self.get_parameter('mjpeg_quality').value)),
            '--flush', '1',
            '-o', '-',
        ]
        if self.get_parameter('hflip').value:
            cmd.append('--hflip')
        if self.get_parameter('vflip').value:
            cmd.append('--vflip')
        shutter = int(self.get_parameter('shutter_us').value)
        if shutter > 0:
            cmd += ['--shutter', str(shutter)]
        gain = float(self.get_parameter('gain').value)
        if gain > 0:
            cmd += ['--gain', str(gain)]
        try:
            return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=0,
                                    env=self._system_env())
        except FileNotFoundError:
            self.get_logger().error(
                f'{binname} not found. Try param rpicam_bin:=libcamera-vid')
            raise

    def _reader_loop(self):
        """Continuously read MJPEG bytes, split into JPEG frames, decode."""
        buf = b''
        while self._running and rclpy.ok():
            chunk = self._proc.stdout.read(8192)
            if not chunk:
                self._dead = True
                self.get_logger().error('rpicam-vid stream ended')
                self._log_proc_stderr()
                return
            buf += chunk
            while True:
                s = buf.find(SOI)
                if s < 0:
                    buf = b''
                    break
                e = buf.find(EOI, s + 2)
                if e < 0:
                    buf = buf[s:]          # keep partial frame
                    break
                jpg = buf[s:e + 2]
                buf = buf[e + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._latest = frame

    def _log_proc_stderr(self):
        try:
            err = self._proc.stderr.read() if self._proc.stderr else b''
            if err:
                self.get_logger().error(
                    'rpicam-vid stderr: ' + err.decode(errors='ignore').strip())
        except Exception:
            pass

    def tick(self):
        with self._lock:
            frame = self._latest
        if frame is None:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self._running = False
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=2)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CsiCameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
