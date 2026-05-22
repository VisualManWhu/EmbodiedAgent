"""Laptop-side GPU object detector + semantic-SLAM mapper for the Pi5 robot.

Polls the Pi's /snapshot endpoint for the newest frame, runs YOLOv8 on the
laptop GPU (RTX 4060), POSTs detection results back to the Pi's
det_bridge_node, shows an annotated preview window locally, and serves the
latest annotated frame over HTTP at /annotated so the Telegram bot can send
detection-boxed photos to the phone.

With --slam it additionally builds a semantic map: AprilTag localization
(slam.pose_estimator) + multi-view-fused object landmarks (slam.semantic_map)
from the same frames. The map is rendered as a top-down PNG (served at :8091)
and POSTed to the Pi's semantic_map_node for RViz. The detection pipeline runs
unchanged whether or not --slam is set.

Snapshot polling (not an MJPEG stream) keeps latency low — there is no stream
pipe to buffer up; every fetch returns the current frame.

Pure Python — no ROS install needed on the laptop. Windows native Python 3.11.

Setup (Windows PowerShell):
    pip install ultralytics opencv-python requests numpy pyyaml
    pip install pupil-apriltags                       # only for --slam
    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Run:
    python laptop_detector.py --pi 172.20.10.4
    python laptop_detector.py --pi 172.20.10.4 --slam

Stop: press q in the preview window, or Ctrl+C.
"""
import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# latest annotated (detection-boxed) JPEG, shared with the HTTP server thread
_annotated_lock = threading.Lock()
_annotated_jpg = None


def _start_annotated_server(port):
    """Serve the latest annotated frame as a single JPEG at GET /annotated."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if not self.path.startswith('/annotated'):
                self.send_response(404)
                self.end_headers()
                return
            with _annotated_lock:
                jpg = _annotated_jpg
            if jpg is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpg)))
            self.end_headers()
            try:
                self.wfile.write(jpg)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class SlamRunner:
    """Per-frame semantic-SLAM: AprilTag pose + object fusion + map outputs.

    Isolated so the YOLO detection path is untouched. Periodically renders the
    top-down map, POSTs it to the Pi semantic_map_node, and persists the map.
    """

    RENDER_INTERVAL = 1.0
    POST_INTERVAL = 1.0
    SAVE_INTERVAL = 15.0

    def __init__(self, pi_ip, semantic_port, map_port, tag_map_path,
                 intrinsics_path, map_json_path, cam_height_m,
                 fusion_overrides=None, nav_overrides=None):
        from slam import camera_calib
        from slam.apriltag_detector import AprilTagDetector
        from slam.pose_estimator import PoseEstimator, load_tag_map
        from slam.semantic_map import FusionConfig, SemanticMap
        from slam.map_renderer import MapRenderer, MapImageServer

        self.K, self.dist, _ = camera_calib.load_intrinsics(intrinsics_path)
        self.tag_map = load_tag_map(tag_map_path)
        cfg = FusionConfig()
        for k, v in (fusion_overrides or {}).items():
            if v is not None:
                setattr(cfg, k, v)
        print(f'fusion: gate={cfg.gate_radius_m}m  '
              f'confirm>={cfg.confirm_min_obs}obs/{cfg.confirm_min_viewpoints}views  '
              f'stale={cfg.tentative_stale_sec}s  '
              f'miss={cfg.miss_demote}/{cfg.miss_prune}')
        self.tag_detector = AprilTagDetector()
        mount = {'forward_m': 0.0, 'lateral_m': 0.0, 'height_m': cam_height_m}
        self.pose_est = PoseEstimator(self.tag_map, self.K, self.dist, mount)

        self.map_json_path = map_json_path
        if os.path.exists(map_json_path):
            self.smap = SemanticMap.load(map_json_path, cfg)
            print(f'semantic map: loaded {len(self.smap.landmarks)} landmarks '
                  f'from {map_json_path}')
        else:
            self.smap = SemanticMap(cfg)

        self.renderer = MapRenderer()
        self.map_server = MapImageServer(map_port)
        self.nav_api_port = map_port + 1            # default 8092
        self._nav_done_lock = threading.Lock()
        self._nav_done_event = None
        self.post_url = f'http://{pi_ip}:{semantic_port}/map'
        self.tags_payload = [
            {'id': i, 'x': e['x'], 'y': e['y'], 'z': e.get('z', 0.0)}
            for i, e in self.tag_map.items()]

        from slam.dead_reckoner import DeadReckoner
        from slam.nav_session import NavConfig, NavSession   # noqa: F401

        nav_overrides = nav_overrides or {}
        # dead-reckoner caps double as the "no-tag grace" window for the
        # NavSession scan trigger — pose_stale fires once either cap trips.
        dr_time = (nav_overrides.get('dr_time_limit_sec')
                   or nav_overrides.get('no_tag_grace_sec') or 5.0)
        dr_dist = nav_overrides.get('dr_distance_limit_m') or 0.5
        self.dead_reckoner = DeadReckoner(time_limit_sec=dr_time,
                                          distance_limit_m=dr_dist)
        self.nav_config = NavConfig()
        nav_field_map = {
            'arrived_radius_m': 'arrived_radius_m',
            'v_max': 'nav_v_max',
            'w_max': 'nav_w_max',
            'w_eff': 'nav_w_eff',
            'max_pulse_sec': 'max_pulse_sec',
            'block_retries': 'block_retries',
            'scan_max_rotations': 'scan_max_rotations',
        }
        for cfg_field, override_key in nav_field_map.items():
            val = nav_overrides.get(override_key)
            if val is not None:
                setattr(self.nav_config, cfg_field, val)
        print(f'nav: arrived={self.nav_config.arrived_radius_m}m  '
              f'v={self.nav_config.v_max}m/s  '
              f'w_cmd={self.nav_config.w_max}/w_eff={self.nav_config.w_eff}rad/s  '
              f'pulse<={self.nav_config.max_pulse_sec}s  '
              f'dr<={dr_time}s/{dr_dist}m  '
              f'block_retries={self.nav_config.block_retries}  '
              f'scan_rot<={self.nav_config.scan_max_rotations}')
        # used by Task 7 nav api when filtering class candidates.
        self.landmark_conf_min = nav_overrides.get('landmark_conf_min', 0.7)
        self.session_nav = None                     # NavSession | None
        self.event_url = f'http://{pi_ip}:9091/status'
        self.cmd_url = f'http://{pi_ip}:9091/command'
        self.http = requests.Session()
        self.last_robot = None
        self._last_render = 0.0
        self._last_post = 0.0
        self._last_save = 0.0
        self._post_ok = None
        # start the nav api LAST so its handler threads can never see partly
        # initialized state (landmark_conf_min, last_robot, session_nav, etc.)
        self._start_nav_api()
        print(f'semantic SLAM: {len(self.tag_map)} tags, map served at '
              f'http://0.0.0.0:{map_port}/map.png')

    def process(self, frame, detections, pan_rad, tilt_rad, now):
        """Fuse one frame and tick any active goto.

        Returns ``(n_tags_seen, located, nav_status)`` where ``nav_status`` is
        the NavSession state string (or None if no goto is active).
        """
        tag_dets = self.tag_detector.detect(frame)
        pose = self.pose_est.estimate(tag_dets, pan_rad, tilt_rad)
        located = pose is not None
        if located:
            T_map_cam = pose['T_map_cam']
            h, w = frame.shape[:2]
            for det in detections:
                self.smap.add_detection(det, T_map_cam, pose['base_xytheta'],
                                        self.K, now)
            self.smap.end_frame(T_map_cam, self.K, (w, h), now)
            rx, ry, ryaw = pose['base_xytheta']
            self.renderer.update_path(rx, ry)
            self.last_robot = (rx, ry, ryaw)

        if now - self._last_render >= self.RENDER_INTERVAL:
            self._render()
            self._last_render = now
        if now - self._last_post >= self.POST_INTERVAL:
            self._post()
            self._last_post = now
        if now - self._last_save >= self.SAVE_INTERVAL:
            self.save()
            self._last_save = now

        # --- nav update --------------------------------------------------
        nav_status = None
        if self.session_nav is not None:
            if located:
                rx, ry, ryaw = pose['base_xytheta']
                self.dead_reckoner.reset((rx, ry, ryaw), now)
                self.session_nav.on_tag_fix(now)
                robot_pose, pose_stale = (rx, ry, ryaw), False
            else:
                robot_pose, pose_stale = self.dead_reckoner.pose_at(now)
            target_ahead = self._target_seen_ahead(detections, frame.shape)
            cmd = self.session_nav.tick(robot_pose, now, pose_stale=pose_stale,
                                        target_ahead=target_ahead)
            if cmd is not None and cmd.kind != 'stop':
                self._send_nav(cmd, now)
            self._drain_pi_events()
            # ARRIVED-on-first-tick (cmd.kind == 'stop') and FAILED('lost')
            # from scan exhaustion reach terminal state WITHOUT a Pi event,
            # so _drain_pi_events would never publish them — handle directly.
            if (self.session_nav is not None
                    and self.session_nav.state.value in ('ARRIVED', 'FAILED')):
                self._publish_nav_done()
            if self.session_nav is not None:
                nav_status = self.session_nav.state.value
        return len(tag_dets), located, nav_status

    def _target_seen_ahead(self, detections, frame_shape):
        """True if YOLO currently sees the active goal's object class large
        and centred — i.e. the thing right in front is the goal, not an
        obstacle. Returns False for tag goals (no COCO class to match)."""
        if self.session_nav is None or not self.session_nav.goal_label:
            return False
        h, w = frame_shape[:2]
        for d in detections:
            if d['cls'] != self.session_nav.goal_label:
                continue
            cx_frac = d['cx'] / w
            h_frac = d['h'] / h
            if 0.30 < cx_frac < 0.70 and h_frac > 0.40:
                return True
        return False

    def _render(self):
        robot = self.last_robot
        img = self.renderer.render(self.tags_payload,
                                   self.smap.snapshot(), robot)
        self.map_server.set_image(img)

    def _post(self):
        robot = None
        if self.last_robot is not None:
            robot = {'x': self.last_robot[0], 'y': self.last_robot[1],
                     'yaw': self.last_robot[2]}
        payload = {'robot': robot, 'tags': self.tags_payload,
                   'landmarks': self.smap.snapshot()}
        try:
            self.http.post(self.post_url, json=payload, timeout=0.5)
            if self._post_ok is not True:
                print('semantic map POST to Pi: OK')
                self._post_ok = True
        except requests.RequestException:
            if self._post_ok is not False:
                print('warning: cannot reach Pi semantic_map_node')
                self._post_ok = False

    def save(self):
        try:
            self.smap.save(self.map_json_path)
        except OSError as e:
            print(f'warning: could not save semantic map: {e}')

    def _start_nav_api(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import json as _json
        runner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == '/nav/done':
                    with runner._nav_done_lock:
                        ev = runner._nav_done_event
                        runner._nav_done_event = None
                    body = _json.dumps(ev or {}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == '/nav/landmarks':
                    runner._handle_landmarks_list(self)
                elif self.path == '/nav/tags':
                    runner._handle_tags_list(self)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/nav/candidates':
                    runner._handle_candidates(self)
                elif self.path == '/nav/goto':
                    runner._handle_goto(self)
                elif self.path == '/nav/stop':
                    runner.stop_goto()
                    self.send_response(200)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

        server = ThreadingHTTPServer(('127.0.0.1', self.nav_api_port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f'nav api at http://127.0.0.1:{self.nav_api_port}/')

    def _handle_candidates(self, h):
        import json as _json
        import types as _types
        from slam.landmark_selector import by_class
        n = int(h.headers.get('Content-Length', 0))
        body = _json.loads(h.rfile.read(n))
        cls = body['target_class']
        robot = self.last_robot or (0.0, 0.0, 0.0)
        # snapshot the landmarks dict on the server thread (atomic in CPython)
        # so the main thread is free to mutate self.smap.landmarks during
        # `process()` without tripping the iteration in by_class.
        smap_view = _types.SimpleNamespace(landmarks=dict(self.smap.landmarks))
        cands = by_class(smap_view, cls, (robot[0], robot[1]),
                         conf_min=self.landmark_conf_min)
        out = [{'id': lm.id, 'label': lm.label,
                'x': float(lm.position[0]), 'y': float(lm.position[1]),
                'dist_m': float(((lm.position[0]-robot[0])**2
                                 + (lm.position[1]-robot[1])**2) ** 0.5)}
               for lm in cands]
        resp = _json.dumps({'candidates': out}).encode()
        h.send_response(200)
        h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(resp)))
        h.end_headers()
        h.wfile.write(resp)

    def _handle_goto(self, h):
        import json as _json
        from slam.landmark_selector import by_id, by_tag
        n = int(h.headers.get('Content-Length', 0))
        req = _json.loads(h.rfile.read(n))
        if 'landmark_id' in req:
            lm = by_id(self.smap, req['landmark_id'])
            if lm is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto((float(lm.position[0]), float(lm.position[1])),
                            lm.id, lm.label)
        elif 'tag_id' in req:
            xy = by_tag(self.tag_map, req['tag_id'])
            if xy is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto(xy, None, f"tag{req['tag_id']}")
        elif 'xy' in req:
            self.start_goto(tuple(req['xy']), None, req.get('label', 'point'))
        else:
            h.send_response(400); h.end_headers(); return
        h.send_response(200); h.end_headers()

    def _handle_landmarks_list(self, h):
        import json as _json
        snap = self.smap.snapshot(confirmed_only=True)
        body = _json.dumps({'landmarks': snap}).encode()
        h.send_response(200)
        h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def _handle_tags_list(self, h):
        import json as _json
        tags = [{'id': i, 'x': float(e['x']), 'y': float(e['y'])}
                for i, e in self.tag_map.items()]
        body = _json.dumps({'tags': tags}).encode()
        h.send_response(200)
        h.send_header('Content-Type', 'application/json')
        h.send_header('Content-Length', str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    # ---- navigation lifecycle ------------------------------------------

    def start_goto(self, goal_xy, goal_id, goal_label):
        from slam.nav_session import NavSession
        # NavSession imported lazily so SlamRunner can be constructed without
        # the slam package; only used when --slam, which already imported it.
        self.session_nav = NavSession(goal_xy=goal_xy, goal_id=goal_id,
                                      goal_label=goal_label,
                                      config=self.nav_config)
        print(f'NAV start -> {goal_label} id={goal_id} at {goal_xy}')

    def stop_goto(self):
        if self.session_nav is not None:
            print('NAV stop')
            try:
                self.http.post(self.cmd_url, json={'action': 'nav_stop'},
                               timeout=0.5)
            except Exception:                       # noqa: BLE001
                pass
            self.session_nav = None

    def _send_nav(self, cmd, now):
        payload = {'action': 'nav_pulse',
                   'vx': cmd.vx, 'vy': cmd.vy, 'wz': cmd.wz,
                   'seconds': cmd.seconds}
        try:
            self.http.post(self.cmd_url, json=payload, timeout=0.5)
        except Exception:                           # noqa: BLE001
            # Pi unreachable — release the WAITING_PULSE latch so the next
            # tick retries instead of stalling forever for a pulse_done that
            # will never arrive.
            from slam.nav_session import State
            if (self.session_nav is not None
                    and self.session_nav.state is State.WAITING_PULSE):
                self.session_nav.state = State.DRIVING
            return
        self.dead_reckoner.record_pulse(cmd.vx, cmd.vy, cmd.wz,
                                        cmd.seconds, now)

    def _drain_pi_events(self):
        if self.session_nav is None:
            return
        try:
            r = self.http.get(self.event_url, timeout=0.5)
            if r.status_code != 200:
                return
            event = r.json().get('event') or ''
        except Exception:                           # noqa: BLE001
            return
        if event:
            self.session_nav.on_pi_event(event)
            if self.session_nav.state.value in ('ARRIVED', 'FAILED'):
                self._publish_nav_done()

    def _publish_nav_done(self):
        reason = (self.session_nav.fail_reason
                  if self.session_nav.state.value == 'FAILED' else None)
        ev = {'event': 'nav_done',
              'state': self.session_nav.state.value,
              'goal_label': self.session_nav.goal_label,
              'goal_id': self.session_nav.goal_id,
              'reason': reason}
        print(f'NAV result: {ev}')
        with self._nav_done_lock:
            self._nav_done_event = ev
        if self.session_nav.state.value == 'FAILED':
            try:
                self.http.post(self.cmd_url,
                               json={'action': 'nav_stop'}, timeout=0.5)
            except Exception:
                pass
        self.session_nav = None


def main():
    global _annotated_jpg

    ap = argparse.ArgumentParser()
    ap.add_argument('--pi', required=True, help='Raspberry Pi IP address')
    ap.add_argument('--stream-port', type=int, default=8080)
    ap.add_argument('--bridge-port', type=int, default=9090)
    ap.add_argument('--annotated-port', type=int, default=8090,
                    help='local port serving annotated frames to the bot')
    ap.add_argument('--model', default='yolov8x.pt')  # strongest YOLOv8; RTX 4060 8GB runs it fine, best classification
    ap.add_argument('--conf', type=float, default=0.4)  # cut low-confidence misclassifications (car<->airplane flicker)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help="GPU index, or 'cpu'")
    ap.add_argument('--no-show', action='store_true', help='disable preview window')
    # semantic SLAM
    _slam_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slam')
    ap.add_argument('--slam', action='store_true',
                    help='build a semantic map from AprilTags + detections')
    ap.add_argument('--semantic-port', type=int, default=9092,
                    help='Pi semantic_map_node port')
    ap.add_argument('--map-port', type=int, default=8091,
                    help='local port serving the top-down map PNG')
    ap.add_argument('--tag-map', default=os.path.join(_slam_dir, 'tag_map.yaml'))
    ap.add_argument('--intrinsics',
                    default=os.path.join(_slam_dir, 'camera_intrinsics.yaml'))
    ap.add_argument('--map-json',
                    default=os.path.join(_slam_dir, 'semantic_map.json'))
    ap.add_argument('--cam-height', type=float, default=0.15,
                    help='camera height above the floor (m), pan/tilt centred')
    # fusion tuning — leave None to keep SemanticMap defaults
    ap.add_argument('--gate-radius', type=float, default=None,
                    help='detection<->landmark association radius (m)')
    ap.add_argument('--confirm-min-obs', type=int, default=None,
                    help='detections needed to CONFIRM a landmark')
    ap.add_argument('--confirm-min-viewpoints', type=int, default=None,
                    help='distinct viewpoints needed to CONFIRM')
    ap.add_argument('--tentative-stale-sec', type=float, default=None,
                    help='prune TENTATIVE landmarks idle this long (s)')
    ap.add_argument('--miss-demote', type=int, default=None,
                    help='in-FOV misses that demote a CONFIRMED landmark')
    ap.add_argument('--miss-prune', type=int, default=None,
                    help='in-FOV misses that prune any landmark')
    # navigation
    ap.add_argument('--arrived-radius-m', type=float, default=0.4)
    ap.add_argument('--nav-v-max', type=float, default=0.15)
    ap.add_argument('--nav-w-max', type=float, default=0.5,
                    help='commanded yaw rate for rotate/scan pulses')
    ap.add_argument('--nav-w-eff', type=float, default=1.12,
                    help='measured real yaw rate at --nav-w-max '
                         '(90deg / rotate_90_sec); sets rotate pulse duration')
    ap.add_argument('--max-pulse-sec', type=float, default=10.0)
    ap.add_argument('--no-tag-grace-sec', type=float, default=5.0)
    ap.add_argument('--dr-distance-limit-m', type=float, default=1.5)
    ap.add_argument('--dr-time-limit-sec', type=float, default=10.0)
    ap.add_argument('--block-retries', type=int, default=3)
    ap.add_argument('--scan-max-rotations', type=int, default=4)
    ap.add_argument('--landmark-conf-min', type=float, default=0.7)
    args = ap.parse_args()

    snapshot_url = f'http://{args.pi}:{args.stream_port}/snapshot'
    bridge_url = f'http://{args.pi}:{args.bridge_port}/detections'

    print(f'loading model {args.model} on device {args.device} ...')
    model = YOLO(args.model)
    names = model.names

    _start_annotated_server(args.annotated_port)
    print(f'annotated frames served at http://0.0.0.0:{args.annotated_port}/annotated')

    slam = None
    if args.slam:
        fusion_overrides = {
            'gate_radius_m': args.gate_radius,
            'confirm_min_obs': args.confirm_min_obs,
            'confirm_min_viewpoints': args.confirm_min_viewpoints,
            'tentative_stale_sec': args.tentative_stale_sec,
            'miss_demote': args.miss_demote,
            'miss_prune': args.miss_prune,
        }
        nav_overrides = {
            'arrived_radius_m': args.arrived_radius_m,
            'nav_v_max': args.nav_v_max,
            'nav_w_max': args.nav_w_max,
            'nav_w_eff': args.nav_w_eff,
            'max_pulse_sec': args.max_pulse_sec,
            'no_tag_grace_sec': args.no_tag_grace_sec,
            'dr_distance_limit_m': args.dr_distance_limit_m,
            'dr_time_limit_sec': args.dr_time_limit_sec,
            'block_retries': args.block_retries,
            'scan_max_rotations': args.scan_max_rotations,
            'landmark_conf_min': args.landmark_conf_min,
        }
        try:
            slam = SlamRunner(args.pi, args.semantic_port, args.map_port,
                              args.tag_map, args.intrinsics, args.map_json,
                              args.cam_height, fusion_overrides, nav_overrides)
        except (ImportError, FileNotFoundError, KeyError) as e:
            print(f'semantic SLAM disabled: {e}')
            slam = None

    session = requests.Session()
    posted_ok = False
    stream_ok = False
    last_log = time.time()
    frames = 0

    print(f'polling frames from {snapshot_url}')
    print(f'posting detections to {bridge_url}')
    print('running — press q in the window (or Ctrl+C) to stop')
    try:
        while True:
            # --- fetch newest frame ---
            try:
                r = session.get(snapshot_url, timeout=2.0)
            except requests.RequestException:
                if stream_ok:
                    print('warning: lost connection to Pi camera stream')
                    stream_ok = False
                time.sleep(0.3)
                continue
            if r.status_code != 200:
                time.sleep(0.1)
                continue
            if not stream_ok:
                print('camera stream: OK')
                stream_ok = True

            buf = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            # pan-tilt angles for this frame (mjpeg_node /snapshot headers)
            pan_yaw = float(r.headers.get('X-Pan-Yaw', 0.0) or 0.0)
            pan_tilt = float(r.headers.get('X-Pan-Tilt', 0.0) or 0.0)

            # --- detect ---
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]
            dets = []
            if res.boxes is not None:
                for b in res.boxes:
                    cid = int(b.cls.item())
                    name = (names[cid] if isinstance(names, list)
                            else names.get(cid, str(cid)))
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append({
                        'cls': name,
                        'score': float(b.conf.item()),
                        'cx': (x1 + x2) / 2.0,
                        'cy': (y1 + y2) / 2.0,
                        'w': max(1.0, x2 - x1),
                        'h': max(1.0, y2 - y1),
                    })

            # --- annotated frame: preview window + /annotated endpoint ---
            annotated = res.plot()
            ok, jpg = cv2.imencode('.jpg', annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _annotated_lock:
                    _annotated_jpg = jpg.tobytes()

            # --- post results back to Pi ---
            try:
                session.post(bridge_url,
                             json={'width': w, 'height': h, 'detections': dets},
                             timeout=0.5)
                if not posted_ok:
                    print('detection POST to Pi: OK')
                    posted_ok = True
            except requests.RequestException:
                if posted_ok:
                    print('warning: lost connection to Pi det_bridge')
                    posted_ok = False

            # --- semantic SLAM (optional, isolated from the above) ---
            slam_info = ''
            if slam is not None:
                try:
                    n_tags, located, nav_state = slam.process(
                        frame, dets, pan_yaw, pan_tilt, time.time())
                    slam_info = (f'  tags={n_tags} '
                                 f'{"LOCATED" if located else "no-fix"}'
                                 f'  nav={nav_state or "-"}')
                except Exception as e:  # noqa: BLE001 - never kill detection
                    slam_info = f'  slam-error: {e}'

            if not args.no_show:
                cv2.imshow('laptop detector (q=quit)', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frames += 1
            now = time.time()
            if now - last_log >= 5.0:
                print(f'fps={frames / (now - last_log):.1f}  '
                      f'last_dets={len(dets)}{slam_info}')
                frames = 0
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        if slam is not None:
            slam.save()
            print('semantic map saved')
        cv2.destroyAllWindows()
        print('stopped')


if __name__ == '__main__':
    main()
