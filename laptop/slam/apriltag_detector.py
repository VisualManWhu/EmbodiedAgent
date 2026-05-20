"""AprilTag detection — thin wrapper over the ``pupil-apriltags`` library.

Install (laptop): ``pip install pupil-apriltags``

The library import is deferred to construction time so the rest of the SLAM
package (and its tests) import fine without the library present.
"""
import cv2
import numpy as np


class AprilTagDetector:
    """Detects ``tag36h11`` markers and returns their image corners.

    ``detect()`` yields one dict per tag: ``{'id', 'corners', 'center'}``.
    ``corners`` is a 4x2 pixel array in the order bottom-left, bottom-right,
    top-right, top-left — matching ``pose_estimator.tag_object_points``.

    pupil-apriltags emits corners cyclically shifted by 2 relative to that
    order; left uncorrected, solvePnP recovers a pose rotated 180 deg about the
    optical axis. ``detect()`` rolls the corners by 2 to align them. This is
    verified by ``tests/test_slam.py::test_real_apriltag_corner_order_and_pose``.
    """

    def __init__(self, families: str = 'tag36h11', nthreads: int = 2,
                 quad_decimate: float = 1.0):
        try:
            from pupil_apriltags import Detector
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                'pupil-apriltags not installed — run: pip install pupil-apriltags'
            ) from e
        self._det = Detector(families=families, nthreads=nthreads,
                             quad_decimate=quad_decimate)

    def detect(self, frame: np.ndarray) -> list:
        gray = (frame if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        out = []
        for d in self._det.detect(gray):
            corners = np.roll(np.asarray(d.corners, dtype=np.float64),
                              2, axis=0)
            out.append({
                'id': int(d.tag_id),
                'corners': corners,
                'center': (float(d.center[0]), float(d.center[1])),
            })
        return out
