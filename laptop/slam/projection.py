"""Monocular geometry: pixels to world rays, ground-plane projection, and
multi-view triangulation. Pure numpy — no hardware or ROS dependency.
"""
import numpy as np

from . import transforms


def pixel_to_ray_cam(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Unit ray direction for pixel ``(u, v)`` in the camera optical frame.

    Points out of the lens (optical +z forward).
    """
    Kinv = np.linalg.inv(np.asarray(K, dtype=float))
    d = Kinv @ np.array([u, v, 1.0])
    return d / np.linalg.norm(d)


def pixel_ray_in_world(T_world_cam: np.ndarray, u: float, v: float,
                       K: np.ndarray):
    """World-frame ray for a pixel. Returns ``(origin, direction)`` where
    ``origin`` is the camera centre and ``direction`` is a unit vector."""
    d_cam = pixel_to_ray_cam(u, v, K)
    origin = T_world_cam[:3, 3].copy()
    direction = transforms.apply_direction(T_world_cam, d_cam)
    direction = direction / np.linalg.norm(direction)
    return origin, direction


def ground_plane_intersect(origin, direction, plane_z: float = 0.0):
    """Intersect a ray with the horizontal plane ``z = plane_z``.

    Returns the 3D world point, or ``None`` if the ray is parallel to the
    plane or points away from it (no forward intersection).
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    dz = direction[2]
    if abs(dz) < 1e-9:
        return None
    t = (plane_z - origin[2]) / dz
    if t <= 0:
        return None
    return origin + t * direction


def triangulate_rays(origins, directions):
    """Least-squares closest point to a set of 3D lines.

    Each line ``i`` is ``origins[i] + s * directions[i]``. Minimises the sum of
    squared perpendicular distances. Needs >= 2 rays. Returns the 3D point, or
    ``None`` if the system is degenerate (e.g. all rays parallel).
    """
    origins = [np.asarray(o, dtype=float).reshape(3) for o in origins]
    directions = [np.asarray(d, dtype=float).reshape(3) for d in directions]
    if len(origins) < 2 or len(origins) != len(directions):
        return None
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, directions):
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        P = np.eye(3) - np.outer(d, d)   # projector onto the line's normal space
        A += P
        b += P @ o
    if np.linalg.cond(A) > 1e12:
        return None
    return np.linalg.solve(A, b)


def world_to_pixel(T_world_cam: np.ndarray, K: np.ndarray, point):
    """Project a world point into the camera image.

    Returns ``(u, v)`` in pixels, or ``None`` if the point is behind the
    camera. Caller checks image bounds.
    """
    point = np.asarray(point, dtype=float).reshape(3)
    p_cam = transforms.apply_point(transforms.invert(T_world_cam), point)
    if p_cam[2] <= 1e-6:
        return None
    uvw = np.asarray(K, dtype=float) @ p_cam
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def viewpoints_distinct(pose_a, pose_b, min_dist_m: float,
                        min_angle_rad: float) -> bool:
    """True if two robot poses count as distinct viewpoints.

    Poses are ``(x, y, yaw)``. Distinct if the camera moved far enough OR
    rotated enough that a second observation adds real triangulation baseline /
    parallax — not just another frame from the same spot.
    """
    ax, ay, ayaw = pose_a
    bx, by, byaw = pose_b
    dist = np.hypot(bx - ax, by - ay)
    dyaw = abs(np.arctan2(np.sin(byaw - ayaw), np.cos(byaw - ayaw)))
    return dist >= min_dist_m or dyaw >= min_angle_rad
