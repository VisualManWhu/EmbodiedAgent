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
_CONFIRMED = (60, 220, 60)
_TENTATIVE = (40, 200, 230)
_TEXT = (240, 240, 240)


class MapRenderer:
    """Accumulates the robot path and renders the top-down map image."""

    def __init__(self, px_per_m: float = 110.0, margin_px: int = 70,
                 min_span_m: float = 3.0):
        self.px_per_m = px_per_m
        self.margin = margin_px
        self.min_span = min_span_m
        self.path: list[tuple[float, float]] = []

    def update_path(self, x: float, y: float, min_step_m: float = 0.05):
        """Append a robot position, skipping near-duplicate points."""
        if self.path:
            px, py = self.path[-1]
            if np.hypot(x - px, y - py) < min_step_m:
                return
        self.path.append((float(x), float(y)))

    def render(self, tags: list, landmarks: list, robot=None) -> np.ndarray:
        """Render to a BGR image. ``tags`` and ``landmarks`` are the dicts from
        ``SemanticMap.snapshot()`` / the tag map; ``robot`` is ``(x, y, yaw)``
        or ``None``."""
        xs, ys = [], []
        for t in tags:
            xs.append(t['x']); ys.append(t['y'])
        for lm in landmarks:
            xs.append(lm['x']); ys.append(lm['y'])
        for (px, py) in self.path:
            xs.append(px); ys.append(py)
        if robot is not None:
            xs.append(robot[0]); ys.append(robot[1])
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

        # robot path
        if len(self.path) >= 2:
            pts = np.array([to_px(x, y) for x, y in self.path], np.int32)
            cv2.polylines(img, [pts], False, _PATH, 2, cv2.LINE_AA)

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
            else:
                cv2.circle(img, (u, v), 10, colour, 2, cv2.LINE_AA)
            label = f"{lm.get('label', '?')} {lm.get('confidence', 0):.0%}"
            cv2.putText(img, label, (u + 13, v + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1, cv2.LINE_AA)

        # robot
        if robot is not None:
            self._draw_robot(img, to_px, robot)

        self._draw_legend(img, len(tags), len(landmarks))
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

    def _draw_robot(self, img, to_px, robot):
        x, y, yaw = robot
        u, v = to_px(x, y)
        nose = (u + int(22 * np.cos(yaw)), v - int(22 * np.sin(yaw)))
        left = (u + int(13 * np.cos(yaw + 2.5)), v - int(13 * np.sin(yaw + 2.5)))
        right = (u + int(13 * np.cos(yaw - 2.5)), v - int(13 * np.sin(yaw - 2.5)))
        cv2.fillConvexPoly(img, np.array([nose, left, right], np.int32), _ROBOT)

    def _draw_legend(self, img, n_tags, n_landmarks):
        cv2.putText(img, f'tags:{n_tags}  objects:{n_landmarks}  (1 grid = 1 m)',
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT, 1,
                    cv2.LINE_AA)


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
