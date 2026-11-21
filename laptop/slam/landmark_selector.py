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


def tag_standoff(tag_map, tag_id, standoff_m: float):
    """Where to stop to LOOK at an AprilTag: ``((x, y), final_yaw)`` — the
    point ``standoff_m`` out along the tag's face normal, heading back at the
    tag. None if the tag is unknown.

    A tag is on a wall, so its own (x, y) is unreachable, and from right
    underneath it the low camera cannot see it at all.
    """
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
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
