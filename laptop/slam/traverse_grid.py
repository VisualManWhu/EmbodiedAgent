"""Traversability grid the robot builds for itself — no configuration beyond
the AprilTags.

A lattice of nodes is laid over the bounding rectangle of the tag positions
(the tags hang on the walls, so that is roughly the room). The rectangle, not
the convex hull: a wall without a tag makes the hull cut the corner off, and
in the field that removed exactly the part of the room that needed mapping.
Nodes that fall outside the real walls simply stay unknown — the robot can
never reach them. Every node is

    UNKNOWN  never visited
    SAFE     the robot has driven over it
    DANGER   the robot was blocked there, or got stuck around it
    NEAR     not dangerous itself, but next to a danger node

and navigation plans over the lattice instead of driving a straight line:
shortest path, with unknown ground costing more than known-safe ground and
danger impassable.

What makes this workable with ~10-30 cm of localization error is that the
evidence is coarse and hard: "the robot was here" marks a 25 cm node, not a
centimetre-accurate obstacle outline. And because the robot can only have got
anywhere over nodes it drove on, there is always a known-safe way back out.

Furniture moves, so danger is not forever: a danger node the robot has since
driven across twice becomes safe again.
"""
import heapq
import json
import math
import os

UNKNOWN = 'unknown'
SAFE = 'safe'
DANGER = 'danger'
NEAR = 'near'

CLEAR_PASSES = 2            # drives across a danger node that clear it

_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]


def convex_hull(points):
    """Convex hull (counter-clockwise) of 2D points — monotone chain."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside(hull, p, eps=1e-9):
    n = len(hull)
    if n < 3:
        return False
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if ((b[0] - a[0]) * (p[1] - a[1])
                - (b[1] - a[1]) * (p[0] - a[0])) < -eps:
            return False
    return True


class TraverseGrid:
    def __init__(self, hull, spacing_m: float = 0.25,
                 unknown_weight: float = 3.0, near_penalty: float = 6.0):
        self.hull = [(float(x), float(y)) for x, y in hull]
        self.spacing = float(spacing_m)
        self.unknown_weight = float(unknown_weight)
        self.near_penalty = float(near_penalty)
        # (i, j) -> {'visits', 'danger', 'clear', 'kind'}
        self.nodes = {}
        self.last_plan = None
        self._last_marked = None
        s = self.spacing
        xs = [p[0] for p in self.hull] or [0.0]
        ys = [p[1] for p in self.hull] or [0.0]
        for i in range(math.floor(min(xs) / s), math.ceil(max(xs) / s) + 1):
            for j in range(math.floor(min(ys) / s),
                           math.ceil(max(ys) / s) + 1):
                if _inside(self.hull, (i * s, j * s)):
                    self.nodes[(i, j)] = {'visits': 0, 'danger': False,
                                          'clear': 0, 'kind': None}

    @classmethod
    def from_tags(cls, tag_map, shape: str = 'rect', **kw):
        """``shape`` — 'rect' (bounding rectangle of the tags, default) or
        'hull' (their convex hull)."""
        pts = [(float(e['x']), float(e['y'])) for e in tag_map.values()]
        if shape == 'hull':
            return cls(convex_hull(pts), **kw)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return cls([(min(xs), min(ys)), (max(xs), min(ys)),
                    (max(xs), max(ys)), (min(xs), max(ys))], **kw)

    # ---- lookup --------------------------------------------------------

    def _key(self, xy):
        """Grid node containing ``xy``, or None if that is off the grid."""
        k = (round(xy[0] / self.spacing), round(xy[1] / self.spacing))
        return k if k in self.nodes else None

    def _nearest_key(self, xy, allow=None):
        """Nearest node to ``xy`` (optionally only those passing ``allow``)."""
        k = self._key(xy)
        if k is not None and (allow is None or allow(k)):
            return k
        best, best_d = None, float('inf')
        for key in self.nodes:
            if allow is not None and not allow(key):
                continue
            d = math.hypot(key[0] * self.spacing - xy[0],
                           key[1] * self.spacing - xy[1])
            if d < best_d:
                best, best_d = key, d
        return best

    def _xy(self, key):
        return (key[0] * self.spacing, key[1] * self.spacing)

    def _danger(self, key):
        return self.nodes[key]['danger']

    def _near(self, key):
        return any(self.nodes.get((key[0] + di, key[1] + dj),
                                  {}).get('danger', False)
                   for di, dj in _NEIGHBOURS)

    def _state(self, key):
        n = self.nodes[key]
        if n['danger']:
            return DANGER
        if self._near(key):
            return NEAR
        return SAFE if n['visits'] > 0 else UNKNOWN

    def state_at(self, xy):
        k = self._key(xy)
        return None if k is None else self._state(k)

    def snapshot(self):
        """Every node as ``{'x', 'y', 'state'}`` — for rendering."""
        return [{'x': k[0] * self.spacing, 'y': k[1] * self.spacing,
                 'state': self._state(k)} for k in sorted(self.nodes)]

    def counts(self):
        out = {SAFE: 0, UNKNOWN: 0, DANGER: 0, NEAR: 0}
        for k in self.nodes:
            out[self._state(k)] += 1
        return out

    # ---- learning ------------------------------------------------------

    def _visit(self, key):
        if key == self._last_marked:
            return                  # standing still is not another pass
        self._last_marked = key
        n = self.nodes[key]
        n['visits'] += 1
        if n['danger']:
            n['clear'] += 1
            if n['clear'] >= CLEAR_PASSES:
                n['danger'], n['clear'], n['kind'] = False, 0, None

    def mark_path(self, a, b):
        """The robot drove from ``a`` to ``b`` (trusted poses): every node
        along the way is somewhere it can go."""
        steps = max(1, int(math.ceil(math.dist(a[:2], b[:2])
                                     / (self.spacing / 2.0))))
        for k in range(steps + 1):
            t = k / steps
            key = self._key((a[0] + (b[0] - a[0]) * t,
                             a[1] + (b[1] - a[1]) * t))
            if key is not None:
                self._visit(key)

    def _set_danger(self, key, kind):
        n = self.nodes[key]
        n['danger'], n['clear'], n['kind'] = True, 0, kind
        if key == self._last_marked:
            self._last_marked = None    # a later pass over it must count

    def mark_danger_at(self, xy, kind='obstacle'):
        key = self._key(xy)
        if key is not None:
            self._set_danger(key, kind)
        return key

    def mark_blocked(self, pose, ahead_m: float = 0.45):
        """The front sensor tripped: the obstacle is AHEAD of the robot, so
        the node in front is marked, not the one the robot stands on.
        ``ahead_m`` is from the robot's centre: ~0.10 m to the sensor plus
        the 0.35 m range at which the Pi reports a block."""
        x, y, yaw = pose
        target = (x + ahead_m * math.cos(yaw), y + ahead_m * math.sin(yaw))
        key = self._key(target)
        if key is None or key == self._key((x, y)):
            return None
        n = self.nodes[key]
        if n['visits'] > 0 and not n['danger']:
            # The robot has DRIVEN over this node. One block report against
            # that is more likely a pose error than a new obstacle, so it
            # takes a second one to overturn it. (Field: blocks reported
            # with a drifting pose painted open, already-driven floor red.)
            n['hits'] = n.get('hits', 0) + 1
            if n['hits'] < 2:
                return None
        n['hits'] = 0
        self._set_danger(key, 'obstacle')
        return self._xy(key)

    def mark_trapped(self, pose, radius_m: float = 0.4):
        """The robot is stuck here: the whole patch is somewhere not to go."""
        x, y = pose[0], pose[1]
        hit = 0
        r = int(math.ceil(radius_m / self.spacing))
        ci, cj = round(x / self.spacing), round(y / self.spacing)
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                if (i, j) in self.nodes and math.hypot(
                        i * self.spacing - x,
                        j * self.spacing - y) <= radius_m + 1e-9:
                    # a sensed obstacle stays an obstacle
                    if self.nodes[(i, j)]['kind'] != 'obstacle':
                        self._set_danger((i, j), 'trapped')
                    hit += 1
        return hit

    # ---- planning ------------------------------------------------------

    ESCAPE_WEIGHT = 10.0        # crossing the danger patch the robot is in

    def _escape_set(self, start):
        """If the robot stands on a danger node: the connected "stuck here"
        patch around it. Those nodes (and only those) may be crossed, at a
        high price, so there is always a way OUT of trouble — but never into
        another danger area, and never through a node where an OBSTACLE was
        actually sensed. (Field bug: the patch marked around a stuck robot
        merged with the obstacle nodes in front of it, and the 'way out' was
        planned straight through the obstacle.)"""
        if start is None or not self._danger(start):
            return frozenset()
        seen, todo = {start}, [start]
        while todo:
            cur = todo.pop()
            for di, dj in _NEIGHBOURS:
                nxt = (cur[0] + di, cur[1] + dj)
                if (nxt in self.nodes and nxt not in seen
                        and self._danger(nxt)
                        and self.nodes[nxt]['kind'] != 'obstacle'):
                    seen.add(nxt)
                    todo.append(nxt)
        return frozenset(seen)

    def _weight(self, key):
        n = self.nodes[key]
        if n['danger']:
            return self.ESCAPE_WEIGHT
        w = 1.0 if n['visits'] > 0 else self.unknown_weight
        if self._near(key):
            w += self.near_penalty
        return w

    def _segment(self, a, b, escape=frozenset()):
        """Cost of driving straight from ``a`` to ``b`` and the unknown
        length on the way, or None if that crosses danger / leaves the grid.
        Danger nodes in ``escape`` (the patch the robot is standing in) may
        be crossed, so a robot in trouble can still leave."""
        length = math.dist(a, b)
        steps = max(1, int(math.ceil(length / (self.spacing / 4.0))))
        step = length / steps
        cost = unknown = 0.0
        for k in range(1, steps + 1):
            t = k / steps
            key = self._key((a[0] + (b[0] - a[0]) * t,
                             a[1] + (b[1] - a[1]) * t))
            if key is None:
                return None
            if self._danger(key) and key not in escape:
                return None
            cost += step * self._weight(key)
            if self.nodes[key]['visits'] == 0:
                unknown += step
        return cost, unknown

    def plan(self, start_xy, goal_xy, goal_radius_m: float = 0.0):
        """Waypoints from ``start_xy`` toward ``goal_xy`` (the start
        excluded), or None when the goal cannot be reached without crossing
        danger. Minimises length weighted by how well the ground is known;
        ``self.last_plan`` records length and unknown fraction.

        With ``goal_radius_m`` = 0 the route ends at the exact goal. With a
        radius it ends at the cheapest-to-reach node within that distance of
        the goal — "anywhere close enough counts". A goal is often a spot
        beside a wall (a tag standoff in a corner): its own nearest node can
        be IN the wall, and insisting on it sent the robot along the wall
        line again and again in the field.
        """
        start_xy = (float(start_xy[0]), float(start_xy[1]))
        goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        start = self._nearest_key(start_xy)
        self.last_plan = None
        if start is None:
            return None
        escape = self._escape_set(start)
        if goal_radius_m > 0.0:
            targets = {k for k in self.nodes
                       if (not self._danger(k) or k == start)
                       and math.dist(self._xy(k), goal_xy) <= goal_radius_m}
            if not targets:
                return None
            # heuristic anchor: the acceptable node nearest the goal
            goal = min(targets, key=lambda k: math.dist(self._xy(k), goal_xy))
        else:
            goal = self._nearest_key(goal_xy)
            if goal is None or (self._danger(goal) and goal != start):
                return None
            targets = {goal}
        slack = goal_radius_m * 2.0     # keeps the heuristic admissible

        def blocked(k):
            return self._danger(k) and k not in escape

        def h(k):
            return max(0.0, math.hypot(k[0] - goal[0], k[1] - goal[1])
                       * self.spacing - slack)

        best = {start: 0.0}
        came = {}
        heap = [(h(start), 0.0, start)]
        while heap:
            _, g, cur = heapq.heappop(heap)
            if cur in targets:
                goal = cur
                break
            if g > best.get(cur, float('inf')):
                continue
            for di, dj in _NEIGHBOURS:
                nxt = (cur[0] + di, cur[1] + dj)
                if nxt not in self.nodes or blocked(nxt):
                    continue
                if di and dj:
                    # no squeezing diagonally between two danger nodes
                    a, b = (cur[0] + di, cur[1]), (cur[0], cur[1] + dj)
                    if ((a in self.nodes and blocked(a))
                            or (b in self.nodes and blocked(b))):
                        continue
                ng = g + (math.hypot(di, dj) * self.spacing
                          * self._weight(nxt))
                if ng < best.get(nxt, float('inf')):
                    best[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(heap, (ng + h(nxt), ng, nxt))
        if goal not in targets or goal not in best:
            return None

        keys = [goal]
        while keys[-1] != start:
            keys.append(came[keys[-1]])
        keys.reverse()
        # Drive from the robot's real position to the real goal, through the
        # lattice nodes in between.
        if goal_radius_m > 0.0:
            pts = [start_xy] + [self._xy(k) for k in keys[1:]]
            if len(pts) == 1:                   # already close enough
                pts.append(start_xy)
        else:
            pts = [start_xy] + [self._xy(k) for k in keys[1:-1]] + [goal_xy]
        pts = self._straighten(pts, escape)
        total = unknown = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = self._segment(a, b, escape)
            total += math.dist(a, b)
            unknown += seg[1] if seg is not None else 0.0
        self.last_plan = {
            'length_m': total,
            'unknown_frac': (unknown / total) if total > 1e-9 else 0.0,
            'waypoints': len(pts) - 1}
        return pts[1:]

    def _straighten(self, pts, escape):
        """Drop waypoints wherever driving straight is just as good: a
        node-by-node path would stop the robot every 25 cm. A shortcut is
        taken only if it crosses no danger and is no costlier than the route
        it replaces — so a deliberate detour over known-safe ground is kept.
        """
        seg = [self._segment(a, b, escape) for a, b in zip(pts, pts[1:])]
        prefix = [0.0]
        for s, (a, b) in zip(seg, zip(pts, pts[1:])):
            # a lattice step that the sampler dislikes (e.g. the goal lies
            # just off the grid) still has to be driven: price it by length
            prefix.append(prefix[-1] + (s[0] if s is not None
                                        else math.dist(a, b)))
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                direct = self._segment(pts[i], pts[j], escape)
                if (direct is not None
                        and direct[0] <= (prefix[j] - prefix[i]) + 1e-9):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ---- persistence ---------------------------------------------------

    def _signature(self):
        return {'spacing': self.spacing,
                'hull': [[round(x, 3), round(y, 3)] for x, y in self.hull]}

    def save(self, path):
        data = {'signature': self._signature(),
                'nodes': [[k[0], k[1], n['visits'], int(n['danger']),
                           n['clear'], n['kind']]
                          for k, n in sorted(self.nodes.items())
                          if n['visits'] or n['danger']]}
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def load(self, path) -> bool:
        """Restore what was learned. False (and nothing loaded) if the file is
        missing or was saved for a different tag layout / spacing."""
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        if data.get('signature') != self._signature():
            return False
        for i, j, visits, danger, clear, kind in data.get('nodes', []):
            if (i, j) in self.nodes:
                self.nodes[(i, j)] = {'visits': int(visits),
                                      'danger': bool(danger),
                                      'clear': int(clear), 'kind': kind}
        return True
