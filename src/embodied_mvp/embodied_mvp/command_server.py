"""Thread-safe HTTP command/status server for agent_node.

POST /command  -- JSON body stored as the latest pending command
GET  /status   -- returns
                    {"mode": ..., "boot_id": ..., "event_seq": N,
                     "events": [{"seq": n, "event": "..."}, ...],
                     "event": "<latest event or ''>"}

Events are NOT consumed by a GET. Every event gets a monotonically increasing
sequence number and the last EVENT_HISTORY of them are returned on each
/status, so any number of readers (telegram bot, laptop detector, a browser, a
debugging curl) can poll without stealing events from each other. Each reader
remembers the highest seq it has handled and ignores anything at or below it.
`boot_id` changes when the server restarts, telling readers the seq counter
started over.

agent_node polls take_command() each control tick, calls set_mode() every
tick, and calls post_event() once when a notable event occurs.
"""
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EVENT_HISTORY = 32


class CommandServer:
    def __init__(self, port):
        self._lock = threading.Lock()
        self._pending = None
        self._mode = 'IDLE'
        self._boot_id = f'{time.time():.3f}'
        self._seq = 0
        self._events = collections.deque(maxlen=EVENT_HISTORY)
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
            self._seq += 1
            self._events.append({'seq': self._seq, 'event': event})

    def status(self):
        with self._lock:
            events = list(self._events)
            return {
                'mode': self._mode,
                'boot_id': self._boot_id,
                'event_seq': self._seq,
                'events': events,
                'event': events[-1]['event'] if events else '',
            }

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
                body = json.dumps(server.status()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler
