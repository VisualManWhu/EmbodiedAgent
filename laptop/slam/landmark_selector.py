"""Pick which landmark a class-level goto refers to.

Pure functions over ``SemanticMap`` — no side effects, no state.
"""
import math

from .semantic_map import CONFIRMED


def by_id(smap, lid):
    """Look up a landmark by id, or return None."""
    return smap.landmarks.get(int(lid))


def by_class(smap, cls, robot_xy, conf_min: float = 0.7,
             state: str | None = CONFIRMED):
    """Landmarks matching ``cls``, filtered by state/confidence, nearest first.

    Ties on distance break deterministically by ascending id.
    """
    candidates = []
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    for lm in smap.landmarks.values():
        if state is not None and lm.state != state:
            continue
        if lm.label != cls:
            continue
        if lm.confidence < conf_min:
            continue
        d = math.hypot(lm.position[0] - rx, lm.position[1] - ry)
        candidates.append((d, lm.id, lm))
    candidates.sort(key=lambda t: (t[0], t[1]))
    return [lm for _, _, lm in candidates]


def nearest_by_class(smap, cls, robot_xy, **kw):
    """Single nearest matching landmark, or None."""
    cands = by_class(smap, cls, robot_xy, **kw)
    return cands[0] if cands else None


# The camera sits ~15 cm off the floor, pitched slightly down, and cannot tilt
# up: the top of its view is only about this far above horizontal. A tag
# mounted higher than the camera therefore drops out of view when the robot
# gets close. Field data: tag 0 (top edge 0.36 m) is just visible from 1.0 m
# (11.6 deg); tag 5 (top edge 0.59 m) was never seen from 1.4 m (17 deg).
CAM_UP_FOV_RAD = math.radians(12.0)


def min_view_distance(entry, cam_height_m: float,
                      up_fov_rad: float = CAM_UP_FOV_RAD) -> float:
    """Closest horizontal distance from which the whole tag is in view."""
    top = float(entry.get('z', 0.0)) + float(entry.get('size_m', 0.0)) / 2.0
    rise = top - cam_height_m
    if rise <= 0.0:
        return 0.0
    return rise / math.tan(up_fov_rad)


def tag_standoff(tag_map, tag_id, standoff_m: float,
                 cam_height_m: float | None = None):
    """Where to stop to LOOK at an AprilTag: ``((x, y), final_yaw)`` — the
    point ``standoff_m`` out along the tag's face normal, heading back at the
    tag. None if the tag is unknown.

    A tag is on a wall, so its own (x, y) is unreachable, and from right
    underneath it the low camera cannot see it at all. With ``cam_height_m``
    the distance is stretched for high-mounted tags until the tag fits in
    view (plus 10% margin).
    """
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    if cam_height_m is not None:
        standoff_m = max(standoff_m,
                         1.1 * min_view_distance(e, cam_height_m))
    normal = math.radians(float(e.get('yaw_deg', 0.0)))
    x = float(e['x']) + standoff_m * math.cos(normal)
    y = float(e['y']) + standoff_m * math.sin(normal)
    facing = math.atan2(-math.sin(normal), -math.cos(normal))
    return (x, y), facing


def by_tag(tag_map, tag_id):
    """``(x, y)`` of an AprilTag from the tag map, or None."""
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    return (float(e['x']), float(e['y']))
