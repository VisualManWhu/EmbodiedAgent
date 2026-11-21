"""Top-down semantic-map renderer + HTTP server.

Draws the AprilTag anchors, the robot path, and the labelled object landmarks
(CONFIRMED solid, TENTATIVE translucent) as a 2D PNG. ``MapImageServer`` serves
the latest render at ``http://<laptop>:8091/map.png`` so the Telegram bot can
fetch it for the phone.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# BGR colours
_BG = (40, 40, 40)
_GRID = (60, 60, 60)
_TAG = (255, 110, 60)
_PATH = (120, 200, 120)
_ROBOT = (0, 120, 255)
# Object landmarks are magenta: green / yellow / red / blue are taken by the
# traversability nodes, and the old green dot read as a big "safe" node.
_CONFIRMED = (230, 70, 230)
_TENTATIVE = (215, 150, 215)
_TEXT = (240, 240, 240)
_ROUTE = (255, 255, 255)
_ROBOT_STALE = (150, 150, 150)      # pose too old to trust
_PATH_BREAK_M = 0.8                 # a jump this big is not driving
_PATH_MAX_POINTS = 800
# traversability grid nodes
_NODE = {'danger': (40, 40, 235),       # red
         'near': (0, 215, 255),         # yellow: next to danger
         'unknown': (235, 140, 50),     # blue
         'safe': (70, 200, 70)}         # green


class MapRenderer:
    """Accumulates the robot path and renders the top-down map image."""

    def __init__(self, px_per_m: float = 110.0, margin_px: int = 70,
                 min_span_m: float = 3.0):
        self.px_per_m = px_per_m
        self.margin = margin_px
        self.min_span = min_span_m
        # trail of fixes; None marks a break (the robot was carried, or a
        # fix was wrong) so no line is drawn across the jump
        self.path: list = []

    def update_path(self, x: float, y: float, min_step_m: float = 0.05):
        """Append a robot position, skipping near-duplicate points. A jump
        too large to have been driven starts a new stretch instead of
        drawing a line across the room."""
        last = next((p for p in reversed(self.path) if p is not None), None)
        if last is not None:
            d = np.hypot(x - last[0], y - last[1])
            if d < min_step_m:
                return
            if d > _PATH_BREAK_M and self.path[-1] is not None:
                self.path.append(None)
        self.path.append((float(x), float(y)))
        if len(self.path) > _PATH_MAX_POINTS:
            del self.path[:len(self.path) - _PATH_MAX_POINTS]

    def render(self, tags: list, landmarks: list, robot=None, grid=None,
               route=None, robot_source: str = 'fix', goal=None,
               status=None) -> np.ndarray:
        """Render to a BGR image. ``tags`` and ``landmarks`` are the dicts from
        ``SemanticMap.snapshot()`` / the tag map; ``robot`` is ``(x, y, yaw)``
        or ``None``. ``grid`` is ``TraverseGrid.snapshot()`` — nodes drawn
        red (danger), yellow (next to danger), blue (unknown), green (safe).
        ``route`` is the planned polyline ``[(x, y), ...]`` if one is active.
        ``robot_source`` says how good the robot pose is: 'fix' (a tag fix,
        drawn solid), 'dr' (dead reckoned since the last fix, drawn hollow)
        or 'stale' (too old to trust, drawn grey). ``goal`` is
        ``(x, y, label)`` of the active goto; ``status`` a list of text lines
        for the bottom of the image.
        """
        # The view is the ROOM: the tags (on its walls) and the grid laid
        # over them. Everything else is drawn inside that frame and simply
        # falls off the edge if it lies outside. Sizing the image to fit
        # every landmark let a few mislocated ones (tens of metres away)
        # stretch it to 1518 x 4351 px with the room a speck in the middle.
        xs, ys = [], []
        for n in grid or ():
            xs.append(n['x']); ys.append(n['y'])
        for t in tags:
            xs.append(t['x']); ys.append(t['y'])
        if not xs:
            # no room to frame (no tags known): fit whatever there is
            for lm in landmarks:
                xs.append(lm['x']); ys.append(lm['y'])
            for p in self.path:
                if p is not None:
                    xs.append(p[0]); ys.append(p[1])
            if robot is not None:
                xs.append(robot[0]); ys.append(robot[1])
            if goal is not None:
                xs.append(goal[0]); ys.append(goal[1])
        if not xs:
            xs, ys = [0.0], [0.0]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # enforce a minimum span so a near-empty map is not absurdly zoomed
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        span_x = max(max_x - min_x, self.min_span)
        span_y = max(max_y - min_y, self.min_span)
        min_x, max_x = cx - span_x / 2, cx + span_x / 2
        min_y, max_y = cy - span_y / 2, cy + span_y / 2

        W = int(span_x * self.px_per_m) + 2 * self.margin
        H = int(span_y * self.px_per_m) + 2 * self.margin
        img = np.full((H, W, 3), _BG, np.uint8)

        def to_px(wx, wy):
            u = int((wx - min_x) * self.px_per_m) + self.margin
            v = H - (int((wy - min_y) * self.px_per_m) + self.margin)
            return u, v

        self._draw_grid(img, min_x, max_x, min_y, max_y, to_px)

        # traversability nodes (underneath everything else)
        for n in grid or ():
            cv2.circle(img, to_px(n['x'], n['y']), 5,
                       _NODE.get(n['state'], _NODE['unknown']), -1,
                       cv2.LINE_AA)

        # robot path — one polyline per unbroken stretch
        stretch = []
        for p in list(self.path) + [None]:
            if p is None:
                if len(stretch) >= 2:
                    pts = np.array([to_px(x, y) for x, y in stretch], np.int32)
                    cv2.polylines(img, [pts], False, _PATH, 2, cv2.LINE_AA)
                stretch = []
            else:
                stretch.append(p)

        # planned route, on top of the trail: from the robot through each
        # waypoint to the goal
        if route and len(route) >= 2:
            pts = np.array([to_px(x, y) for x, y in route], np.int32)
            cv2.polylines(img, [pts], False, _ROUTE, 2, cv2.LINE_AA)
            for u, v in pts[1:-1]:
                cv2.circle(img, (int(u), int(v)), 5, _ROUTE, 1, cv2.LINE_AA)
        if goal is not None:
            u, v = to_px(goal[0], goal[1])
            cv2.circle(img, (u, v), 9, _ROUTE, 2, cv2.LINE_AA)
            cv2.circle(img, (u, v), 3, _ROUTE, -1, cv2.LINE_AA)
            if len(goal) > 2 and goal[2]:
                cv2.putText(img, f'goal: {goal[2]}', (u + 13, v - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _ROUTE, 1,
                            cv2.LINE_AA)

        # tags
        for t in tags:
            u, v = to_px(t['x'], t['y'])
            cv2.rectangle(img, (u - 9, v - 9), (u + 9, v + 9), _TAG, -1)
            cv2.putText(img, f"T{t['id']}", (u + 12, v + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TAG, 1, cv2.LINE_AA)

        # landmarks
        for lm in landmarks:
            u, v = to_px(lm['x'], lm['y'])
            confirmed = lm.get('state') == 'CONFIRMED'
            colour = _CONFIRMED if confirmed else _TENTATIVE
            if confirmed:
                cv2.circle(img, (u, v), 10, colour, -1, cv2.LINE_AA)
                cv2.circle(img, (u, v), 10, _TEXT, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (u, v), 10, colour, 2, cv2.LINE_AA)
            label = f"{lm.get('label', '?')} {lm.get('confidence', 0):.0%}"
            cv2.putText(img, label, (u + 13, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1, cv2.LINE_AA)

        # robot
        if robot is not None:
            self._draw_robot(img, to_px, robot, robot_source)

        self._draw_legend(img, len(tags), len(landmarks), bool(grid))
        if status:
            # a solid strip, so landmark labels near the bottom edge cannot
            # run through the text
            top = H - 14 - 20 * len(status)
            cv2.rectangle(img, (0, top), (W, H), (25, 25, 25), -1)
            cv2.line(img, (0, top), (W, top), _GRID, 1)
            for i, line in enumerate(reversed(status)):
                cv2.putText(img, line, (10, H - 12 - 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1,
                            cv2.LINE_AA)
        return img

    def _draw_grid(self, img, min_x, max_x, min_y, max_y, to_px):
        x0 = np.floor(min_x)
        while x0 <= max_x:
            u, _ = to_px(x0, min_y)
            cv2.line(img, (u, 0), (u, img.shape[0]), _GRID, 1)
            x0 += 1.0
        y0 = np.floor(min_y)
        while y0 <= max_y:
            _, v = to_px(min_x, y0)
            cv2.line(img, (0, v), (img.shape[1], v), _GRID, 1)
            y0 += 1.0

    def _draw_robot(self, img, to_px, robot, source='fix'):
        x, y, yaw = robot
        u, v = to_px(x, y)
        nose = (u + int(22 * np.cos(yaw)), v - int(22 * np.sin(yaw)))
        left = (u + int(13 * np.cos(yaw + 2.5)), v - int(13 * np.sin(yaw + 2.5)))
        right = (u + int(13 * np.cos(yaw - 2.5)), v - int(13 * np.sin(yaw - 2.5)))
        tri = np.array([nose, left, right], np.int32)
        if source == 'fix':
            cv2.fillConvexPoly(img, tri, _ROBOT, cv2.LINE_AA)
        elif source == 'dr':
            # estimated since the last fix: hollow
            cv2.fillConvexPoly(img, tri, _BG, cv2.LINE_AA)
            cv2.polylines(img, [tri], True, _ROBOT, 2, cv2.LINE_AA)
        else:
            cv2.fillConvexPoly(img, tri, _ROBOT_STALE, cv2.LINE_AA)

    def _draw_legend(self, img, n_tags, n_landmarks, with_nodes=False):
        cv2.putText(img, f'tags:{n_tags}  objects:{n_landmarks}  (1 grid = 1 m)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1,
                    cv2.LINE_AA)
        if not with_nodes:
            return
        x = 10
        for state, label in (('safe', 'safe'), ('unknown', 'unknown'),
                             ('near', 'near danger'), ('danger', 'danger')):
            cv2.circle(img, (x + 6, 40), 5, _NODE[state], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x + 16, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, _TEXT, 1, cv2.LINE_AA)
            x += 30 + 9 * len(label)


class MapImageServer:
    """Serves the latest rendered map PNG at GET /map.png (and /)."""

    def __init__(self, port: int = 8091):
        self._lock = threading.Lock()
        self._png = None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                with server._lock:
                    png = server._png
                if png is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(png)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                try:
                    self.wfile.write(png)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self._server.serve_forever,
                         daemon=True).start()

    def set_image(self, bgr: np.ndarray):
        ok, png = cv2.imencode('.png', bgr)
        if ok:
            with self._lock:
                self._png = png.tobytes()

    def shutdown(self):
        try:
            self._server.shutdown()
        except Exception:
            pass
