"""SE(3) rigid-transform helpers shared across the SLAM modules.

Conventions
-----------
- A transform is a 4x4 homogeneous matrix ``T``. ``T_a_b`` is the pose of frame
  ``b`` expressed in frame ``a``: a point ``p_b`` in frame ``b`` maps to frame
  ``a`` as ``p_a = T_a_b @ [p_b; 1]``.
- ``map`` / world frame: x-y on the floor, z up. Robot ``base`` frame: x
  forward, y left, z up, origin on the floor at the robot centre.
- OpenCV optical camera frame: z forward (out of the lens), x right, y down.

All angles are radians.
"""
import cv2
import numpy as np

# optical-frame axes expressed in robot-style axes (columns = optical x,y,z):
#   optical +z (forward) -> robot +x (forward)
#   optical +x (right)   -> robot -y (right, since robot +y is left)
#   optical +y (down)    -> robot -z (down)
R_ROBOT_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_transform(R: np.ndarray, t) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def translation(t) -> np.ndarray:
    return make_transform(np.eye(3), t)


def from_rvec_tvec(rvec, tvec) -> np.ndarray:
    """4x4 transform from an OpenCV Rodrigues rotation vector + translation."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return make_transform(R, np.asarray(tvec, dtype=float).reshape(3))


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (cheaper and more stable than np.linalg)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_point(T: np.ndarray, p) -> np.ndarray:
    """Transform a single 3-point through T."""
    p = np.asarray(p, dtype=float).reshape(3)
    return (T[:3, :3] @ p) + T[:3, 3]


def apply_direction(T: np.ndarray, d) -> np.ndarray:
    """Rotate a single 3-direction through T (ignores translation)."""
    d = np.asarray(d, dtype=float).reshape(3)
    return T[:3, :3] @ d


def pose_xytheta(T: np.ndarray):
    """Extract a planar pose ``(x, y, yaw)`` from a transform.

    ``yaw`` is the heading of the frame's x-axis in the x-y plane.
    """
    x, y = float(T[0, 3]), float(T[1, 3])
    yaw = float(np.arctan2(T[1, 0], T[0, 0]))
    return x, y, yaw


def camera_pose_in_base(pan_rad: float, tilt_rad: float, mount: dict) -> np.ndarray:
    """Pose of the camera optical frame in the robot base frame (``T_base_cam``).

    The camera sits on a 2-DOF pan-tilt: ``pan`` rotates about the base z-axis,
    ``tilt`` about the (panned) y-axis. ``mount`` gives the pan-tilt pivot
    offset from the base origin: keys ``forward_m``, ``lateral_m``, ``height_m``.
    A positive ``tilt`` pitches the camera downward.
    """
    pivot = translation([mount.get('forward_m', 0.0),
                         mount.get('lateral_m', 0.0),
                         mount.get('height_m', 0.0)])
    pan = make_transform(rot_z(pan_rad), [0, 0, 0])
    tilt = make_transform(rot_y(tilt_rad), [0, 0, 0])
    optical = make_transform(R_ROBOT_FROM_OPTICAL, [0, 0, 0])
    return pivot @ pan @ tilt @ optical
