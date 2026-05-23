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


def by_tag(tag_map, tag_id):
    """``(x, y)`` of an AprilTag from the tag map, or None."""
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    return (float(e['x']), float(e['y']))
