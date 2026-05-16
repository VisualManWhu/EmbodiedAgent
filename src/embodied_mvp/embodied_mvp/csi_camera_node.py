"""CSI camera node for Raspberry Pi 5.

RoboStack's conda libcamera can't enumerate the Pi5 CSI camera (missing PiSP
IPA modules). Workaround: spawn the SYSTEM `rpicam-vid` binary (which uses the
working system libcamera), read its raw YUV420 stdout stream, decode with
OpenCV inside the conda env, and publish sensor_msgs/Image.

No python/libcamera bindings are imported here — only a subprocess pipe — so
there is no ABI clash with the conda numpy/opencv.

Params:
  rpicam_bin   : binary name (rpicam-vid, fallback libcamera-vid)
  width/height : capture resolution
  framerate    : fps
  hflip/vflip  : image flips
  frame_id     : header frame_id
"""
import os
import subprocess
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CsiCameraNode(Node):
    def __init__(self):
        super().__init__('camera')

        self.declare_parameter('rpicam_bin', 'rpicam-vid')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('framerate', 15)
        self.declare_parameter('hflip', False)
        self.declare_parameter('vflip', False)
        self.declare_parameter('frame_id', 'camera_link')
        # Exposure: short shutter freezes motion blur while the car moves.
        # 0 = auto. Indoors try 6000-10000 us (1/165 - 1/100 s).
        self.declare_parameter('shutter_us', 0)
        self.declare_parameter('gain', 0.0)        # 0 = auto analogue gain

        self.w = int(self.get_parameter('width').value)
        self.h = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('framerate').value)
        self.frame_id = self.get_parameter('frame_id').value
        # YUV420 (I420): Y plane h*w + U,V each (h/2)*(w/2) -> total w*h*3/2
        self.frame_bytes = self.w * self.h * 3 // 2

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 5)

        self._proc = self._start_rpicam()
        # Read frames in a wall-clock timer; pulls whatever is buffered.
        self.create_timer(1.0 / max(1, self.fps), self.tick)
        self.get_logger().info(
            f'camera (CSI via rpicam-vid) {self.w}x{self.h}@{self.fps} ready')

    def _system_env(self):
        """Env for the rpicam-vid subprocess: strip conda paths so the SYSTEM
        binary loads SYSTEM libs (conda libstdc++/libcamera would crash it)."""
        env = os.environ.copy()
        conda = env.get('CONDA_PREFIX', '')
        # Drop conda from LD_LIBRARY_PATH; system ld.so.cache resolves the rest.
        ld = env.get('LD_LIBRARY_PATH', '')
        if ld:
            kept = [p for p in ld.split(':') if p and (not conda or conda not in p)]
            if kept:
                env['LD_LIBRARY_PATH'] = ':'.join(kept)
            else:
                env.pop('LD_LIBRARY_PATH', None)
        env.pop('LD_PRELOAD', None)
        # Ensure system bin dirs are on PATH to find rpicam-vid.
        env['PATH'] = '/usr/bin:/bin:/usr/local/bin:' + env.get('PATH', '')
        return env

    def _start_rpicam(self):
        binname = self.get_parameter('rpicam_bin').value
        cmd = [
            binname, '-t', '0', '--nopreview',
            '--width', str(self.w), '--height', str(self.h),
            '--framerate', str(self.fps),
            '--codec', 'yuv420',
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
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=0,
                                    env=self._system_env())
        except FileNotFoundError:
            self.get_logger().error(
                f'{binname} not found. Try param rpicam_bin:=libcamera-vid')
            raise
        return proc

    def _log_proc_stderr(self):
        try:
            err = self._proc.stderr.read() if self._proc.stderr else b''
            if err:
                self.get_logger().error(
                    'rpicam-vid stderr: ' + err.decode(errors='ignore').strip())
        except Exception:
            pass

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def tick(self):
        if getattr(self, '_dead', False):
            return
        if self._proc.poll() is not None:
            self._dead = True
            self.get_logger().error('rpicam-vid exited; camera stream dead')
            self._log_proc_stderr()
            return
        raw = self._read_exact(self.frame_bytes)
        if raw is None:
            self._dead = True
            self.get_logger().warn('camera stream EOF')
            self._log_proc_stderr()
            return
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(self.h * 3 // 2, self.w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)

    def destroy_node(self):
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
