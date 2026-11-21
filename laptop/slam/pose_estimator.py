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
    """Load ``tag_map.yaml`` into ``{int id: {x, y, z, yaw_deg, size_m}}``.

    If a sibling ``tag_map.local.yaml`` exists it is preferred — real room
    measurements live in the gitignored ``.local`` file so the tracked
    ``tag_map.yaml`` stays as a template.
    """
    import os as _os
    local = _os.path.join(_os.path.dirname(path), 'tag_map.local.yaml')
    if _os.path.exists(local):
        path = local
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


# A tag's orientation is recovered from how its square is foreshortened. Seen
# face-on there is almost no foreshortening to measure, so the heading it
# yields is ill-conditioned — and its reprojection error is still tiny,
# because many different orientations fit the four corners equally well.
# Field data, robot stationary: one tag at ~5 deg incidence gave headings
# scattered over 10 deg; tags at 54-63 deg were steady to 0.1 deg.
YAW_RELIABLE_INCIDENCE_RAD = np.radians(20.0)
_INCIDENCE_SATURATION_RAD = np.radians(60.0)    # more oblique = no better


def tag_incidence_rad(T_cam_tag: np.ndarray) -> float:
    """Viewing angle of a tag: 0 = seen face-on, pi/2 = seen edge-on."""
    normal = T_cam_tag[:3, 2]
    to_cam = -T_cam_tag[:3, 3]
    n = np.linalg.norm(to_cam)
    if n < 1e-9:
        return 0.0
    return float(np.arccos(np.clip(normal @ (to_cam / n), -1.0, 1.0)))


def _tag_weight(incidence_rad: float, err_px: float) -> float:
    """How much to trust one tag's pose: oblique and well-fitted is best. The
    small floors keep a lone face-on tag usable (its POSITION is still good)
    and stop a near-zero reprojection error from dominating everything."""
    oblique = np.sin(min(incidence_rad, _INCIDENCE_SATURATION_RAD))
    return float((oblique + 0.05) / (err_px + 0.2))


# The camera rides at a fixed height above the floor. A solved pose that puts
# it well above or below that is not a noisy fix, it is a wrong one — the
# mirror-image solution of the planar pose problem, or a far, tiny tag. Field
# data: wrong solutions sat at 0.30-0.86 m, right ones at 0.10-0.21 m (true
# 0.15 m).
CAM_HEIGHT_TOL_M = 0.12


def estimate_camera_pose(tag_detections, tag_map, K, dist=None, details=None,
                         cam_height_m=None, height_tol_m=CAM_HEIGHT_TOL_M):
    """Camera pose ``T_map_cam`` from detected tags with known map poses.

    ``tag_detections`` — list of ``{'id': int, 'corners': 4x2 pixels}``.
    ``tag_map`` — ``{id: {x, y, z, yaw_deg, size_m}}``.
    Returns ``(T_map_cam, reproj_error_px, n_tags_used)`` or ``None`` if no
    detected tag has a known map pose / all solves fail.

    If ``details`` is a dict it is filled with ``incidence_rad`` (viewing
    angle of the tag the orientation was taken from), ``rotation_tag`` and
    ``n_rejected``.

    With ``cam_height_m`` given, a tag whose solution puts the camera more
    than ``height_tol_m`` away from that height is discarded as impossible.
    """
    K = np.asarray(K, dtype=np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)

    estimates = []   # (T_map_cam, error, weight, incidence, tag id)
    rejected = 0
    for det in tag_detections:
        entry = tag_map.get(det['id'])
        if entry is None:
            continue
        solved = _solve_single(det['corners'], entry['size_m'], K, dist)
        if solved is None:
            continue
        T_cam_tag, err = solved
        T_map_cam = tag_pose_in_map(entry) @ transforms.invert(T_cam_tag)
        if (cam_height_m is not None
                and abs(T_map_cam[2, 3] - cam_height_m) > height_tol_m):
            rejected += 1
            continue
        inc = tag_incidence_rad(T_cam_tag)
        estimates.append((T_map_cam, err, _tag_weight(inc, err), inc,
                          det['id']))

    if details is not None:
        details['n_rejected'] = rejected
    if not estimates:
        return None

    # Orientation from the most trustworthy tag — the obliquely viewed one,
    # NOT merely the one with the lowest reprojection error.
    estimates.sort(key=lambda e: -e[2])
    best_T, best_err, _, best_inc, best_id = estimates[0]
    if details is not None:
        details['incidence_rad'] = best_inc
        details['rotation_tag'] = best_id
    if len(estimates) == 1:
        return best_T, best_err, 1

    # translation: mean weighted the same way.
    weights = np.array([e[2] for e in estimates])
    weights /= weights.sum()
    t = sum(w * e[0][:3, 3] for w, e in zip(weights, estimates))
    fused = best_T.copy()
    fused[:3, 3] = t
    mean_err = float(np.mean([e[1] for e in estimates]))
    return fused, mean_err, len(estimates)


class PoseEstimator:
    """Stateful wrapper: adds the camera-mount transform so it can also report
    the robot *base* pose, given the current pan/tilt angles."""

    # A fix that puts the robot this far outside the rectangle spanned by
    # the tags is wrong: the tags hang on the walls, the robot is in the
    # room. (Field: fixes at y = -30 m planted "landmarks" out there.)
    ROOM_MARGIN_M = 0.75

    def __init__(self, tag_map: dict, K, dist=None, mount: dict | None = None):
        self.tag_map = tag_map
        self.rejected_outside = 0
        xs = [float(e['x']) for e in tag_map.values()]
        ys = [float(e['y']) for e in tag_map.values()]
        # needs at least a triangle of tags to mean anything
        self.room = ((min(xs), min(ys), max(xs), max(ys))
                     if len(tag_map) >= 3 else None)
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = (np.zeros(5) if dist is None
                     else np.asarray(dist, dtype=np.float64))
        self.mount = mount or {'forward_m': 0.0, 'lateral_m': 0.0,
                               'height_m': 0.10}

    def estimate(self, tag_detections, pan_rad=0.0, tilt_rad=0.0):
        """Return a pose dict or ``None`` if no known tag is visible.

        Keys: ``T_map_cam`` (4x4), ``T_map_base`` (4x4), ``base_xytheta``
        ``(x, y, yaw)``, ``reproj_error``, ``n_tags``, ``incidence_rad``
        (viewing angle of the tag the heading came from), ``rotation_tag``,
        and ``yaw_reliable`` — False when that tag was seen too face-on for
        its heading to be trusted (the position is still usable).
        """
        details = {}
        result = estimate_camera_pose(
            tag_detections, self.tag_map, self.K, self.dist, details=details,
            cam_height_m=self.mount.get('height_m'))
        if result is None:
            return None
        T_map_cam, err, n = result
        T_base_cam = transforms.camera_pose_in_base(pan_rad, tilt_rad,
                                                    self.mount)
        T_map_base = T_map_cam @ transforms.invert(T_base_cam)
        if self.room is not None:
            bx, by = float(T_map_base[0, 3]), float(T_map_base[1, 3])
            m = self.ROOM_MARGIN_M
            if not (self.room[0] - m <= bx <= self.room[2] + m
                    and self.room[1] - m <= by <= self.room[3] + m):
                self.rejected_outside += 1
                return None
        return {
            'T_map_cam': T_map_cam,
            'T_map_base': T_map_base,
            'base_xytheta': transforms.pose_xytheta(T_map_base),
            'reproj_error': err,
            'n_tags': n,
            'incidence_rad': details['incidence_rad'],
            'rotation_tag': details['rotation_tag'],
            'n_rejected': details['n_rejected'],
            'yaw_reliable': bool(details['incidence_rad']
                                 >= YAW_RELIABLE_INCIDENCE_RAD),
        }
