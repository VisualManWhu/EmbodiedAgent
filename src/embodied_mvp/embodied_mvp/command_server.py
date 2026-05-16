"""Thread-safe HTTP command/status server for agent_node.

POST /command  -- JSON body stored as the latest pending command
GET  /status   -- returns {"mode": ..., "event": ...}; a non-empty event is
                  one-shot, cleared after the first GET that returns it.

agent_node polls take_command() each control tick, calls set_mode() every
tick, and calls post_event() once when a notable event occurs.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class CommandServer:
    def __init__(self, port):
        self._lock = threading.Lock()
        self._pending = None
        self._mode = 'IDLE'
        self._event = ''
        self._server = ThreadingHTTPServer(('0.0.0.0', int(port)),
                                           self._make_handler())
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def take_command(self):
        with self._lock:
            cmd = self._pending
            self._pending = None
            return cmd

    def set_mode(self, mode):
        with self._lock:
            self._mode = mode

    def post_event(self, event):
        with self._lock:
            self._event = event

    def shutdown(self):
        try:
            self._server.shutdown()
        except Exception:
            pass

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not self.path.startswith('/command'):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    cmd = json.loads(self.rfile.read(n))
                    with server._lock:
                        server._pending = cmd
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'ok')
                except Exception as e:  # noqa: BLE001
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(str(e).encode())

            def do_GET(self):
                if not self.path.startswith('/status'):
                    self.send_response(404)
                    self.end_headers()
                    return
                with server._lock:
                    body = json.dumps(
                        {'mode': server._mode, 'event': server._event}).encode()
                    server._event = ''   # one-shot
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
