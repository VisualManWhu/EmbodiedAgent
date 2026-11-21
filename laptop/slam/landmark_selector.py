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


VIEW_MARGIN_M = 0.35                # > the arrival radius of a tag goal


def min_view_distance(entry, cam_height_m: float,
                      up_fov_rad: float = CAM_UP_FOV_RAD) -> float:
    """Closest horizontal distance from which the whole tag is in view."""
    top = float(entry.get('z', 0.0)) + float(entry.get('size_m', 0.0)) / 2.0
    rise = top - cam_height_m
    if rise <= 0.0:
        return 0.0
    return rise / math.tan(up_fov_rad)


# Viewing a tag from up to this far off its face normal is fine (better, in
# fact, for the solved heading); tried in this order when the spot straight
# in front of it is hard against a wall.
STANDOFF_OFFSETS_DEG = (0, 15, -15, 30, -30, 45, -45)
STANDOFF_CLEARANCE_M = 0.5


def tag_bounds(tag_map):
    """Bounding rectangle of the tags ``(xmin, ymin, xmax, ymax)`` — roughly
    the room, the same rectangle the traversability grid covers."""
    xs = [float(e['x']) for e in tag_map.values()]
    ys = [float(e['y']) for e in tag_map.values()]
    return min(xs), min(ys), max(xs), max(ys)


def tag_standoff(tag_map, tag_id, standoff_m: float,
                 cam_height_m: float | None = None, bounds=None,
                 clearance_m: float = STANDOFF_CLEARANCE_M, avoid=None):
    """Where to stop to LOOK at an AprilTag: ``((x, y), final_yaw)`` — a
    point ``standoff_m`` from the tag, heading back at it. None if the tag is
    unknown.

    Normally the point lies straight out along the tag's face normal. With
    ``bounds`` (the room rectangle) that spot must also be ``clearance_m``
    inside the room; if it is not — a tag in a corner has its "straight in
    front" running along the side wall, and the robot cannot get within reach
    of a point 15 cm from a wall — the viewpoint swings round the tag to the
    first offset that is clear. ``avoid(xy) -> bool`` additionally rules
    out spots known to be bad (danger nodes of the traversability grid).

    A tag is on a wall, so its own (x, y) is unreachable, and from right
    underneath it the low camera cannot see it at all. With ``cam_height_m``
    the distance is at least the closest range the whole tag is visible
    from plus ``VIEW_MARGIN_M``: the robot arrives anywhere inside its
    arrival radius of this point, and stopping short of it must still leave
    the tag in view.
    """
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    if cam_height_m is not None:
        standoff_m = max(standoff_m,
                         min_view_distance(e, cam_height_m) + VIEW_MARGIN_M)
    normal = math.radians(float(e.get('yaw_deg', 0.0)))

    def spot(offset_deg):
        a = normal + math.radians(offset_deg)
        return (float(e['x']) + standoff_m * math.cos(a),
                float(e['y']) + standoff_m * math.sin(a)), a

    def clearance(p):
        xmin, ymin, xmax, ymax = bounds
        return min(p[0] - xmin, xmax - p[0], p[1] - ymin, ymax - p[1])

    def bad(p):
        return avoid is not None and avoid(p)

    (x, y), a = spot(0)
    if (bounds is not None and clearance((x, y)) < clearance_m) or bad((x, y)):
        best = None
        for off in STANDOFF_OFFSETS_DEG:
            p, pa = spot(off)
            c = clearance(p) if bounds is not None else float('inf')
            if c >= clearance_m and not bad(p):
                (x, y), a = p, pa
                break
            # fallback ranking: a known-bad spot is worse than a tight one
            score = c - (10.0 if bad(p) else 0.0)
            if best is None or score > best[0]:
                best = (score, p, pa)
        else:
            _, (x, y), a = best         # nowhere is clear: take the roomiest
    return (x, y), math.atan2(-math.sin(a), -math.cos(a))


def by_tag(tag_map, tag_id):
    """``(x, y)`` of an AprilTag from the tag map, or None."""
    e = tag_map.get(int(tag_id))
    if e is None:
        return None
    return (float(e['x']), float(e['y']))
