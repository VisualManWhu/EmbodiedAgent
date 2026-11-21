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
import math
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
        self.cam_height_m = cam_height_m
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
        self.tag_standoff_m = nav_overrides.get('tag_standoff_m') or 1.0
        # Traversability grid the robot learns by driving: nodes it has
        # crossed are safe, nodes where it was blocked / stuck are danger,
        # the rest unknown. Gotos are planned over it instead of driven as a
        # straight line.
        self.grid = None
        self.grid_path = os.path.join(os.path.dirname(map_json_path),
                                      'traverse_grid.json')
        self._prev_fix = None               # (x, y, t) of the last tag fix
        self._last_block = None             # (t, pose) of the last block
        if nav_overrides.get('grid', True):
            from slam.traverse_grid import TraverseGrid
            self.grid = TraverseGrid.from_tags(
                self.tag_map,
                spacing_m=nav_overrides.get('grid_spacing_m') or 0.25,
                unknown_weight=nav_overrides.get('unknown_weight') or 3.0)
            loaded = self.grid.load(self.grid_path)
            self.nav_config.replan_on_block = True
            print(f'traversability grid: {len(self.grid.nodes)} nodes at '
                  f'{self.grid.spacing} m, '
                  f'{"loaded " + str(self.grid.counts()) if loaded else "new"}')
        # Pan-tilt tag aiming: during navigation the camera looks at tags
        # instead of straight ahead, and a lost pose is first recovered by
        # sweeping the camera rather than rotating the chassis.
        self.aimer = None
        self._pan_mismatch_since = None
        self._pan_ignored = 0
        self._last_pan = 0.0
        self._deferred_done = None          # (event, since) awaiting camera
        if nav_overrides.get('pan_aim', True):
            from slam.tag_aimer import AimConfig, TagAimer
            self.aimer = TagAimer(self.tag_map,
                                  AimConfig(cam_height_m=cam_height_m))
            # A camera sweep covers ~200 deg, so when it finds nothing the
            # chassis should turn far, not creep round in 30 deg steps.
            self.nav_config.scan_rotate_rad = math.radians(150.0)
            self.nav_config.scan_max_rotations = 5
            print('pan-tilt tag aiming: on')
        self.session_nav = None                     # NavSession | None
        self.seek = None                            # SeekSession | None
        self._class_hits = {}                       # cls -> deque[timestamp]
        self._class_sight = {}                      # cls -> latest sighting
        from pi_events import EventCursor
        self.pi_events = EventCursor()
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
        self._last_pan = pan_rad
        self._release_deferred_done(now)
        tag_dets = self.tag_detector.detect(frame)
        pose = (self.pose_est.estimate(tag_dets, pan_rad, tilt_rad)
                if self._frame_pan_valid(pan_rad, tag_dets, now) else None)
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
            self._grid_note_fix(rx, ry, now)

        if now - self._last_render >= self.RENDER_INTERVAL:
            self._render()
            self._last_render = now
        if now - self._last_post >= self.POST_INTERVAL:
            self._post()
            self._last_post = now
        if now - self._last_save >= self.SAVE_INTERVAL:
            self.save()
            self._last_save = now

        # --- nav / seek update -------------------------------------------
        self._record_hits(detections, now, frame.shape)
        events = []
        if self.session_nav is not None or self.seek is not None:
            events = self.pi_events.take(self._get_pi_status())
        nav_status = None
        if self.session_nav is not None:
            if located:
                rx, ry, ryaw = pose['base_xytheta']
                self.dead_reckoner.reset((rx, ry, ryaw), now)
                self.session_nav.on_tag_fix(now)
                robot_pose, pose_stale = (rx, ry, ryaw), False
            else:
                robot_pose, pose_stale = self.dead_reckoner.pose_at(now)
            # "the goal object is right ahead" only means something when the
            # camera is looking ahead
            target_ahead = (abs(pan_rad) < 0.09 and self._target_seen_ahead(
                detections, frame.shape))
            # for the NAV-> log line: where nav believes the robot is
            self._nav_pose_dbg = (robot_pose, located, pose_stale)
            self._nav_yaw_ok = bool(located and pose.get('yaw_reliable', True))
            self._nav_fix_tag = pose.get('rotation_tag') if located else None
            hold = self._tick_aimer(now, robot_pose, pose_stale)
            self._plan_route_if_needed(robot_pose, pose_stale)
            cmd = None
            if not hold:
                cmd = self.session_nav.tick(robot_pose, now,
                                            pose_stale=pose_stale,
                                            target_ahead=target_ahead,
                                            pose_fresh=located)
            if cmd is not None and cmd.kind != 'stop':
                self._send_nav(cmd, now)
                if cmd.kind == 'scan_rotate' and self.aimer is not None:
                    self.aimer.reset_sweep()    # new heading: sweep again
            self._feed_nav_events(events, now)
            # ARRIVED-on-first-tick (cmd.kind == 'stop') and FAILED('lost')
            # from scan exhaustion reach terminal state WITHOUT a Pi event,
            # so _feed_nav_events would never publish them — handle directly.
            if (self.session_nav is not None
                    and self.session_nav.state.value in ('ARRIVED', 'FAILED')):
                self._publish_nav_done(now)
            if self.session_nav is not None:
                nav_status = self.session_nav.state.value
                if self.aimer is not None:
                    a = self.aimer
                    nav_status += (
                        f' cam={math.degrees(a.pan_cmd):+.0f}°'
                        + (f'->tag{a.target_id}' if a.target_id is not None
                           else ''))
        seek = self.seek            # local: an api thread may clear it
        if seek is not None:
            self._tick_seek(seek, events, now)
            if not seek.done:
                nav_status = (f'SEEK-{seek.state.value}'
                              + (f'/{nav_status}' if nav_status else ''))
        return len(tag_dets), located, nav_status

    # ---- traversability grid --------------------------------------------

    def _grid_note_fix(self, x, y, now):
        """A tag fix is hard evidence the robot is HERE; two fixes close in
        space and time are evidence it drove the stretch between them. Also
        fed the dead-reckoned pose at the end of each clean forward pulse:
        fixes alone are too sparse (19 nodes after a 160 s drive), and one
        pulse is at most 0.4 m from the last fix."""
        if self.grid is None:
            return
        prev = self._prev_fix
        if (prev is not None and now - prev[2] <= 15.0
                and math.hypot(x - prev[0], y - prev[1]) <= 0.7):
            self.grid.mark_path(prev[:2], (x, y))
        else:
            self.grid.mark_path((x, y), (x, y))
        self._prev_fix = (x, y, now)

    def _plan_route_if_needed(self, robot_pose, pose_stale):
        nav = self.session_nav
        if (self.grid is None or nav is None or robot_pose is None
                or pose_stale or nav.last_tag_t is None):
            return
        if nav.route_ready and not nav.needs_replan:
            return
        # "close enough to the goal" is good enough for the planner too:
        # aim for any reachable node inside the arrival radius
        route = self.grid.plan(
            robot_pose[:2], nav.goal_xy,
            goal_radius_m=0.8 * self.nav_config.arrived_radius_m)
        if route is None:
            from slam.nav_session import State
            print(f'[{time.time():.3f}] PLAN-> no route to '
                  f'({nav.goal_xy[0]:+.2f},{nav.goal_xy[1]:+.2f}) that '
                  f'avoids danger')
            nav.state = State.FAILED
            nav.fail_reason = 'no_path'
            return
        info = self.grid.last_plan
        nav.set_route(route)
        print(f'[{time.time():.3f}] PLAN-> {info["length_m"]:.2f} m, '
              f'{info["unknown_frac"]:.0%} unknown, via '
              + ' '.join(f'({x:+.2f},{y:+.2f})' for x, y in route))

    def _grid_on_block(self, event, now):
        """Record where the robot was blocked (if it was not simply the goal
        it had been sent to)."""
        nav = self.session_nav
        if (self.grid is None or nav is None or event != 'blocked:front'
                or not nav.needs_replan):
            return
        pose, _located, stale = getattr(self, '_nav_pose_dbg',
                                        (None, False, True))
        if pose is None or stale:
            # no trustworthy pose: we know the robot hit something, but not
            # where — marking a guess would poison the grid
            print(f'[{time.time():.3f}] GRID: block ignored (pose not '
                  f'trusted)')
            return
        self._last_block = (now, pose)
        node = self.grid.mark_blocked(pose)
        msg = (f'obstacle at ({node[0]:+.2f},{node[1]:+.2f})'
               if node else 'obstacle off the grid')
        print(f'[{time.time():.3f}] GRID: {msg}')

    def _grid_on_failed(self, nav):
        """A goto that died of repeated blocks: the robot really is stuck
        here, so the patch it is standing in becomes somewhere not to go.
        (Not after a mere three blocks, as first built: feeling its way
        along a wall near the goal tripped that at once and painted the
        open floor the robot was standing on.)"""
        if (self.grid is None or nav.fail_reason != 'blocked'
                or self._last_block is None):
            return
        n = self.grid.mark_trapped(self._last_block[1])
        print(f'[{time.time():.3f}] GRID: goto failed on blocks -> {n} nodes '
              f'around the robot marked danger')

    # ---- pan-tilt tag aiming --------------------------------------------

    def _frame_pan_valid(self, pan_rad, tag_dets, now):
        """False while the camera is swinging to a newly commanded angle: the
        Pi reports the commanded pan immediately, the servo arrives a moment
        later, and a pose solved in between is simply wrong."""
        a = self.aimer
        if a is None:
            return True
        if abs(pan_rad - a.pan_cmd) > 2e-3:
            # Frame predates our command — or the Pi is not following it.
            if self._pan_mismatch_since is None:
                self._pan_mismatch_since = now
            elif now - self._pan_mismatch_since > 2.0:
                self._pan_ignored += 1
                print(f'warning: Pi reports pan {math.degrees(pan_rad):+.0f}° '
                      f'but {math.degrees(a.pan_cmd):+.0f}° was commanded')
                a.reset()
                a.pan_cmd = pan_rad
                self._pan_mismatch_since = None
                if self._pan_ignored >= 3:
                    print('warning: Pi is not following pan-tilt commands — '
                          'tag aiming disabled (deploy the agent_node with '
                          'the `pantilt` command)')
                    self.aimer = None
            return False
        self._pan_mismatch_since = None
        if a.settling(now):
            return False
        self._pan_ignored = 0
        a.note_seen([d['id'] for d in tag_dets if d['id'] in self.tag_map],
                    now)
        return True

    def _tick_aimer(self, now, robot_pose, pose_stale):
        """Point the camera for the active goto. Returns True when the chassis
        must hold still this tick."""
        a, nav = self.aimer, self.session_nav
        if a is None or nav is None:
            return False
        from slam.nav_session import State
        still = nav.state is not State.WAITING_PULSE
        pose_ok = (robot_pose is not None and not pose_stale
                   and nav.last_tag_t is not None)
        pan = a.update(now, robot_pose if pose_ok else None, still)
        if pan is not None:
            print(f'[{time.time():.3f}] AIM-> pan={math.degrees(pan):+5.0f}°  '
                  + (f'tag{a.target_id}' if a.target_id is not None
                     else ('sweep' if not pose_ok else 'ahead')))
            self._post_pi({'action': 'pantilt', 'yaw': pan, 'pitch': 0.0})
        # Hold the chassis (a) while the servo settles — the Pi keeps only ONE
        # pending command, so a nav pulse sent right behind the pan command
        # would overwrite it — and (b) while a lost-pose camera sweep is
        # still under way: look around before turning the whole robot.
        return still and (a.settling(now)
                          or (not pose_ok and not a.sweep_exhausted))

    def _emit_done(self, ev, now):
        """Hand a finished goto / seek to the bot — but not while the camera
        is still pointing at a tag off to the side. The bot photographs the
        instant it sees the result, and in the field that photo showed the
        side wall instead of the goal the robot had just turned to face."""
        if self.aimer is not None and abs(self._last_pan) > 2e-3:
            self._deferred_done = (ev, now)
            return
        with self._nav_done_lock:
            self._nav_done_event = ev

    def _release_deferred_done(self, now):
        if self._deferred_done is None:
            return
        ev, since = self._deferred_done
        centred = abs(self._last_pan) < 2e-3 and now - since >= 0.9
        if centred or now - since > 4.0:        # never hold a result hostage
            self._deferred_done = None
            with self._nav_done_lock:
                self._nav_done_event = ev

    def _recentre_camera(self, send=True):
        """The camera goes back to straight ahead when navigation ends.
        ``send=False`` when a Pi command that re-centres it anyway (`stop`,
        `find`) is about to be sent — two commands in a row would collide in
        the Pi's single pending-command slot."""
        a = self.aimer
        if a is None:
            return
        if send and abs(a.pan_cmd) > 1e-6:
            self._post_pi({'action': 'pantilt', 'yaw': 0.0, 'pitch': 0.0})
        a.reset()

    # ---- class sightings (shared by seek + arrival verification) --------

    HITS_KEPT = 20

    def _record_hits(self, detections, now, frame_shape=None):
        import collections
        biggest = {}
        for d in detections:
            self._class_hits.setdefault(
                d['cls'], collections.deque(maxlen=self.HITS_KEPT)).append(now)
            if 'h' in d and d['h'] > biggest.get(d['cls'], {}).get('h', 0.0):
                biggest[d['cls']] = d
        if frame_shape is None:
            return
        fh = float(frame_shape[0])
        for cls, d in biggest.items():
            top, bottom = d['cy'] - d['h'] / 2.0, d['cy'] + d['h'] / 2.0
            self._class_sight[cls] = {
                't': now, 'h_frac': d['h'] / fh,
                'clipped': top <= 0.02 * fh or bottom >= 0.98 * fh}

    def _hits(self, cls):
        return list(self._class_hits.get(cls, ()))

    def _seen_recently(self, cls, now, window_sec=3.0):
        return any(t >= now - window_sec for t in self._hits(cls))

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
        route = None
        nav = self.session_nav
        if nav is not None and nav.route_ready and robot is not None:
            route = [robot[:2]] + list(nav.waypoints) + [nav.goal_xy]
        img = self.renderer.render(
            self.tags_payload, self.smap.snapshot(), robot,
            grid=self.grid.snapshot() if self.grid is not None else None,
            route=route)
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
        if self.grid is not None:
            try:
                self.grid.save(self.grid_path)
            except OSError as e:
                print(f'warning: could not save traversability grid: {e}')

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
                    runner.seek = None      # an explicit goto ends any seek
                    runner._handle_goto(self)
                elif self.path == '/nav/seek':
                    runner._handle_seek(self)
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
        from slam.landmark_selector import by_id, tag_standoff
        n = int(h.headers.get('Content-Length', 0))
        req = _json.loads(h.rfile.read(n))
        if 'landmark_id' in req:
            lm = by_id(self.smap, req['landmark_id'])
            if lm is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto((float(lm.position[0]), float(lm.position[1])),
                            lm.id, lm.label)
        elif 'tag_id' in req:
            # stop in front of the tag, looking at it — not at the wall it
            # hangs on, where the camera cannot even see it
            spot = tag_standoff(self.tag_map, req['tag_id'],
                                self.tag_standoff_m,
                                cam_height_m=self.cam_height_m)
            if spot is None:
                h.send_response(404); h.end_headers(); return
            self.start_goto(spot[0], None, f"tag{req['tag_id']}",
                            final_yaw=spot[1])
        elif 'xy' in req:
            self.start_goto(tuple(req['xy']), None, req.get('label', 'point'))
        else:
            h.send_response(400); h.end_headers(); return
        h.send_response(200); h.end_headers()

    def _handle_seek(self, h):
        import json as _json
        n = int(h.headers.get('Content-Length', 0))
        req = _json.loads(h.rfile.read(n))
        cls = req.get('target_class')
        if not cls:
            h.send_response(400); h.end_headers(); return
        self.start_seek(cls)
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

    def start_goto(self, goal_xy, goal_id, goal_label, final_yaw=None):
        from slam.nav_session import NavSession
        # NavSession imported lazily so SlamRunner can be constructed without
        # the slam package; only used when --slam, which already imported it.
        # Skip Pi events from before this goto (a previous session's trailing
        # pulse_done / blocked must not be fed to the new NavSession).
        self.pi_events.sync(self._get_pi_status())
        self.session_nav = NavSession(goal_xy=goal_xy, goal_id=goal_id,
                                      goal_label=goal_label,
                                      config=self.nav_config,
                                      final_yaw=final_yaw)
        print(f'[{time.time():.3f}] NAV start -> {goal_label} '
              f'id={goal_id} at {goal_xy}'
              + ('' if final_yaw is None else f' facing {final_yaw:+.2f}rad'))

    def stop_goto(self):
        if self.seek is not None:
            print('SEEK stop')
            self.seek = None
        if self.session_nav is not None:
            print('NAV stop')
            if (self.grid is not None and self._last_block is not None
                    and time.time() - self._last_block[0] < 30.0):
                # stopped by the operator right after being blocked: the
                # strongest "do not go here" signal there is
                n = self.grid.mark_trapped(self._last_block[1])
                print(f'GRID: operator stop after a block -> {n} nodes '
                      f'marked danger')
                self._last_block = None
            # `stop` (not `nav_stop`): it also re-centres the camera on the Pi
            self._post_pi({'action': 'stop'})
            self.session_nav = None
            self._recentre_camera(send=False)

    def _send_nav(self, cmd, now):
        payload = {'action': 'nav_pulse',
                   'vx': cmd.vx, 'vy': cmd.vy, 'wz': cmd.wz,
                   'seconds': cmd.seconds}
        self._last_nav_kind = cmd.kind
        # timestamp is Unix epoch — same clock as the Pi ROS log stamps,
        # so the two logs can be aligned line-for-line.
        print(f'[{time.time():.3f}] NAV-> {cmd.kind:11s} '
              f'vx={cmd.vx:+.2f} vy={cmd.vy:+.2f} wz={cmd.wz:+.2f} '
              f'sec={cmd.seconds:.2f}  state={self.session_nav.state.value}'
              f'  {self._nav_pose_text()}')
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
        # Feed the dead-reckoner the yaw rate the motors REALLY produce, not
        # the commanded wz. In place that is w_eff (~2.2x the command); as
        # steering blended into a forward pulse it is only ~0.6x, because the
        # duty floor swallows most of the left/right difference. Scaling
        # every pulse by the in-place factor made the reckoner believe a
        # steered forward pulse had turned ~80 deg, and nav then "corrected"
        # a turn that never happened.
        from slam.dead_reckoner import modeled_yaw_rate
        yaw_rate = modeled_yaw_rate(cmd.vx, cmd.vy, cmd.wz,
                                    self.nav_config.w_max,
                                    self.nav_config.w_eff)
        self.dead_reckoner.record_pulse(cmd.vx, cmd.vy, yaw_rate,
                                        cmd.seconds, now)

    def _nav_pose_text(self):
        pose, located, stale = getattr(self, '_nav_pose_dbg',
                                       (None, False, True))
        if pose is None:
            return 'pose=none'
        gx, gy = self.session_nav.goal_xy
        # 'tag~' = fix from a tag seen face-on: position good, heading shaky
        src = (('tag' if getattr(self, '_nav_yaw_ok', True) else 'tag~')
               + str(getattr(self, '_nav_fix_tag', '') or '')
               if located else ('stale' if stale else 'dr'))
        return (f'pose=({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}rad) '
                f'[{src}] goal_dist='
                f'{((gx - pose[0]) ** 2 + (gy - pose[1]) ** 2) ** 0.5:.2f}m')

    def _feed_nav_events(self, events, now):
        for event in events:
            if self.session_nav is None:            # stopped mid-drain
                return
            before = self.session_nav.state.value
            self.session_nav.on_pi_event(event)
            self._grid_on_block(event, now)
            if (event == 'pulse_done' and self.grid is not None
                    and getattr(self, '_last_nav_kind', None) == 'forward'):
                pose, _located, stale = getattr(
                    self, '_nav_pose_dbg', (None, False, True))
                if pose is not None and not stale:
                    self._grid_note_fix(pose[0], pose[1], now)
            print(f'[{time.time():.3f}] NAV<- event={event:16s} '
                  f'{before} -> {self.session_nav.state.value}')
            if self.session_nav.state.value in ('ARRIVED', 'FAILED'):
                self._publish_nav_done(now)
                return

    def _get_pi_status(self):
        try:
            r = self.http.get(self.event_url, timeout=0.5)
            return r.json() if r.status_code == 200 else None
        except Exception:                           # noqa: BLE001
            return None

    def _publish_nav_done(self, now):
        nav = self.session_nav
        self.session_nav = None
        failed = nav.state.value == 'FAILED'
        if failed:
            self._post_pi({'action': 'stop'})       # also re-centres the camera
            self._grid_on_failed(nav)
        self._recentre_camera(send=not failed)
        from slam.seek_session import SeekState
        if self.seek is not None and self.seek.state is SeekState.NAV:
            # This goto was a leg of a class seek — the seek decides what the
            # result means (arrived-and-seen vs stale landmark vs retry).
            print(f'[{time.time():.3f}] SEEK nav leg -> {nav.state.value}'
                  f'{" (" + str(nav.fail_reason) + ")" if failed else ""}')
            seek = self.seek
            self._run_seek_actions(seek, seek.on_nav_done(
                nav.state.value, now, self._hits(seek.target_class)))
            return
        # A map arrival is only "really there" if the object is in sight.
        # ``seen`` is None for goals with no object class (tags, raw points).
        seen = (self._seen_recently(nav.goal_label, now)
                if nav.goal_id is not None else None)
        ev = {'event': 'nav_done',
              'state': nav.state.value,
              'goal_label': nav.goal_label,
              'goal_id': nav.goal_id,
              'reason': nav.fail_reason if failed else None,
              'seen': seen}
        print(f'[{time.time():.3f}] NAV result: {ev}')
        self._emit_done(ev, now)

    # ---- class seek: map navigation + visual search as one behaviour ----

    def start_seek(self, target_class):
        from slam.seek_session import SeekSession
        self.session_nav = None
        self._deferred_done = None
        with self._nav_done_lock:
            self._nav_done_event = None
        self.seek = SeekSession(target_class)
        print(f'[{time.time():.3f}] SEEK start -> {target_class}')

    def _seek_candidates(self, cls):
        from slam.landmark_selector import by_class
        robot = self.last_robot or (0.0, 0.0, 0.0)
        return [{'id': lm.id, 'label': lm.label,
                 'x': float(lm.position[0]), 'y': float(lm.position[1])}
                for lm in by_class(self.smap, cls, (robot[0], robot[1]),
                                   conf_min=self.landmark_conf_min)]

    def _post_pi(self, cmd):
        try:
            self.http.post(self.cmd_url, json=cmd, timeout=0.5)
        except Exception:                           # noqa: BLE001
            print(f'warning: could not send {cmd} to Pi')

    def _run_seek_actions(self, seek, actions):
        for act in actions:
            sight = self._class_sight.get(seek.target_class)
            sight_txt = ('never' if sight is None else
                         f"{time.time() - sight['t']:.1f}s ago "
                         f"h={sight['h_frac']:.2f}"
                         f"{' clipped' if sight['clipped'] else ''}")
            print(f'[{time.time():.3f}] SEEK-> {act[0]:10s} '
                  f'state={seek.state.value}  last_sight={sight_txt}  '
                  f'trail={seek.trail}')
            if act[0] == 'find':
                self._post_pi({'action': 'find', 'target': act[1]})
            elif act[0] == 'goto':
                lm = act[1]
                self.start_goto((lm['x'], lm['y']), lm['id'], lm['label'])
            elif act[0] == 'nav_cancel':
                self.session_nav = None     # the next `find` takes the Pi over
                self._recentre_camera(send=False)   # `find` re-centres it
            elif act[0] == 'stop':
                self._post_pi({'action': 'stop'})

    def _tick_seek(self, seek, events, now):
        for event in events:
            seek.on_pi_event(event)
        if not seek.done:
            cls = seek.target_class
            self._run_seek_actions(seek, seek.tick(
                now, self._hits(cls), self._seek_candidates(cls),
                self._class_sight.get(cls)))
        if seek.done:
            arrived = seek.state.value == 'ARRIVED'
            ev = {'event': 'nav_done',
                  'state': seek.state.value,
                  'goal_label': seek.target_class,
                  'goal_id': seek.goal_id if seek.via == 'map' else None,
                  'reason': seek.fail_reason,
                  'seen': True if arrived else None,
                  'via': seek.via,
                  'unseen_ids': list(seek.unseen_ids),
                  'trail': list(seek.trail)}
            print(f'[{time.time():.3f}] SEEK result: {ev}')
            self._emit_done(ev, now)
            if self.seek is seek:
                self.seek = None


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
    ap.add_argument('--nav-w-eff', type=float, default=1.5,
                    help='measured real yaw rate at --nav-w-max '
                         '(90deg / rotate_90_sec); sets rotate pulse duration')
    # keep pulses SHORT: a long forward pulse drives a fixed-curvature arc
    # (blended steering wz x duration) and overruns the dead-reckon caps.
    ap.add_argument('--max-pulse-sec', type=float, default=2.5)
    ap.add_argument('--no-tag-grace-sec', type=float, default=5.0)
    ap.add_argument('--dr-distance-limit-m', type=float, default=0.5)
    ap.add_argument('--dr-time-limit-sec', type=float, default=5.0)
    ap.add_argument('--block-retries', type=int, default=3)
    ap.add_argument('--scan-max-rotations', type=int, default=12,
                    help='scan steps before FAILED(lost); 12 x 30deg = 360deg')
    ap.add_argument('--landmark-conf-min', type=float, default=0.7)
    ap.add_argument('--no-grid', action='store_true',
                    help='drive gotos as straight lines (old behaviour) '
                         'instead of planning over the learned '
                         'traversability grid')
    ap.add_argument('--grid-spacing', type=float, default=0.25,
                    help='traversability grid node spacing (m)')
    ap.add_argument('--unknown-weight', type=float, default=3.0,
                    help='how much costlier unknown ground is than ground '
                         'the robot has already driven (1 = no preference)')
    ap.add_argument('--no-pan-aim', action='store_true',
                    help='keep the camera looking straight ahead during '
                         'navigation (old behaviour) instead of aiming the '
                         'pan-tilt at AprilTags')
    ap.add_argument('--tag-standoff-m', type=float, default=1.0,
                    help='tag goals (回原点/绕室) stop this far in front of '
                         'the tag, facing it, so the tag stays in view')
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
            'tag_standoff_m': args.tag_standoff_m,
            'pan_aim': not args.no_pan_aim,
            'grid': not args.no_grid,
            'grid_spacing_m': args.grid_spacing,
            'unknown_weight': args.unknown_weight,
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
                seen = ','.join(f"{d['cls']}:{d['score']:.2f}" for d in dets)
                print(f'fps={frames / (now - last_log):.1f}  '
                      f'last_dets={len(dets)}'
                      f'{" [" + seen + "]" if seen else ""}{slam_info}')
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
