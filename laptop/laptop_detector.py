"""Laptop-side GPU object detector for the Pi5 search robot.

Polls the Pi's /snapshot endpoint for the newest frame, runs YOLOv8 on the
laptop GPU (RTX 4060), POSTs detection results back to the Pi's
det_bridge_node, shows an annotated preview window locally, and serves the
latest annotated frame over HTTP at /annotated so the Telegram bot can send
detection-boxed photos to the phone.

Snapshot polling (not an MJPEG stream) keeps latency low — there is no stream
pipe to buffer up; every fetch returns the current frame.

Pure Python — no ROS install needed on the laptop. Windows native Python 3.11.

Setup (Windows PowerShell):
    pip install ultralytics opencv-python requests numpy
    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Run:
    python laptop_detector.py --pi 192.168.178.37

Stop: press q in the preview window, or Ctrl+C.
"""
import argparse
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
    args = ap.parse_args()

    snapshot_url = f'http://{args.pi}:{args.stream_port}/snapshot'
    bridge_url = f'http://{args.pi}:{args.bridge_port}/detections'

    print(f'loading model {args.model} on device {args.device} ...')
    model = YOLO(args.model)
    names = model.names

    _start_annotated_server(args.annotated_port)
    print(f'annotated frames served at http://0.0.0.0:{args.annotated_port}/annotated')

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

            if not args.no_show:
                cv2.imshow('laptop detector (q=quit)', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frames += 1
            now = time.time()
            if now - last_log >= 5.0:
                print(f'fps={frames / (now - last_log):.1f}  last_dets={len(dets)}')
                frames = 0
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print('stopped')


if __name__ == '__main__':
    main()
