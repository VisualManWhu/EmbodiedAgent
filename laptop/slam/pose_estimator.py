"""Camera/robot pose in the map frame, solved from observed AprilTags.

Each tag has a known, hand-measured pose in the map frame (``tag_map.yaml``).
``cv2.solvePnP`` recovers the tag's pose in the camera frame from its detected
corners; composing with the known map pose gives the camera pose. With several
tags visible the per-tag estimates are fused (translation = error-weighted
mean, rotation = lowest-reprojection-error tag).
"""
import cv2
import numpy as np
import yaml

from . import transforms


def load_tag_map(path: str) -> dict:
    """Load ``tag_map.yaml`` into ``{int id: {x, y, z, yaw_deg, size_m}}``."""
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data['tags'].items()}


def tag_object_points(size_m: float) -> np.ndarray:
    """The 4 tag corners in the tag's own frame (+x right, +y up, +z out of the
    face), in the canonical order emitted by ``apriltag_detector``:
    bottom-left, bottom-right, top-right, top-left.
    """
    h = size_m / 2.0
    return np.array([
        [-h, -h, 0.0],
        [h, -h, 0.0],
        [h, h, 0.0],
        [-h, h, 0.0],
    ], dtype=np.float64)


def tag_pose_in_map(entry: dict) -> np.ndarray:
    """Transform ``T_map_tag`` from a ``tag_map.yaml`` entry.

    The entry gives the tag-centre position ``x, y, z`` (metres) and ``yaw_deg``
    — the compass direction the tag *face normal* points in the map x-y plane.
    The tag is assumed mounted vertically: tag +y points to map +z (up).
    """
    yaw = np.radians(entry['yaw_deg'])
    c, s = np.cos(yaw), np.sin(yaw)
    # columns = tag x, y, z axes expressed in the map frame
    R = np.array([
        [-s, 0.0, c],
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
    ])
    return transforms.make_transform(R, [entry['x'], entry['y'], entry['z']])


def _solve_single(corners: np.ndarray, size_m: float, K: np.ndarray,
                  dist: np.ndarray):
    """solvePnP for one tag. Returns ``(T_cam_tag, reproj_error_px)`` or None."""
    obj = tag_object_points(size_m)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean((proj.reshape(4, 2) - img) ** 2)))
    return transforms.from_rvec_tvec(rvec, tvec), err


def estimate_camera_pose(tag_detections, tag_map, K, dist=None):
    """Camera pose ``T_map_cam`` from detected tags with known map poses.

    ``tag_detections`` — list of ``{'id': int, 'corners': 4x2 pixels}``.
    ``tag_map`` — ``{id: {x, y, z, yaw_deg, size_m}}``.
    Returns ``(T_map_cam, reproj_error_px, n_tags_used)`` or ``None`` if no
    detected tag has a known map pose / all solves fail.
    """
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)

    estimates = []   # (T_map_cam, error)
    for det in tag_detections:
        entry = tag_map.get(det['id'])
        if entry is None:
            continue
        solved = _solve_single(det['corners'], entry['size_m'], K, dist)
        if solved is None:
            continue
        T_cam_tag, err = solved
        T_map_cam = tag_pose_in_map(entry) @ transforms.invert(T_cam_tag)
        estimates.append((T_map_cam, err))

    if not estimates:
        return None

    estimates.sort(key=lambda e: e[1])
    best_T, best_err = estimates[0]
    if len(estimates) == 1:
        return best_T, best_err, 1

    # translation: inverse-error-weighted mean; rotation: lowest-error tag.
    weights = np.array([1.0 / (e + 1e-6) for _, e in estimates])
    weights /= weights.sum()
    t = sum(w * T[:3, 3] for w, (T, _) in zip(weights, estimates))
    fused = best_T.copy()
    fused[:3, 3] = t
    mean_err = float(np.mean([e for _, e in estimates]))
    return fused, mean_err, len(estimates)


class PoseEstimator:
    """Stateful wrapper: adds the camera-mount transform so it can also report
    the robot *base* pose, given the current pan/tilt angles."""

    def __init__(self, tag_map: dict, K, dist=None, mount: dict | None = None):
        self.tag_map = tag_map
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = (np.zeros(5) if dist is None
                     else np.asarray(dist, dtype=np.float64))
        self.mount = mount or {'forward_m': 0.0, 'lateral_m': 0.0,
                               'height_m': 0.10}

    def estimate(self, tag_detections, pan_rad=0.0, tilt_rad=0.0):
        """Return a pose dict or ``None`` if no known tag is visible.

        Keys: ``T_map_cam`` (4x4), ``T_map_base`` (4x4), ``base_xytheta``
        ``(x, y, yaw)``, ``reproj_error``, ``n_tags``.
        """
        result = estimate_camera_pose(tag_detections, self.tag_map,
                                      self.K, self.dist)
        if result is None:
            return None
        T_map_cam, err, n = result
        T_base_cam = transforms.camera_pose_in_base(pan_rad, tilt_rad,
                                                    self.mount)
        T_map_base = T_map_cam @ transforms.invert(T_base_cam)
        return {
            'T_map_cam': T_map_cam,
            'T_map_base': T_map_base,
            'base_xytheta': transforms.pose_xytheta(T_map_base),
            'reproj_error': err,
            'n_tags': n,
        }
