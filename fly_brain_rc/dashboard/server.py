"""Standard-library HTTP/SSE server; only mission commands cross this boundary."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from queue import Empty
from threading import Event, Lock, Thread


class MissionController:
    def __init__(self, running=False):
        self.lock = Lock()
        self.status = 'running' if running else 'idle'
        self.target_class = 'red_ball'
        self.generation = 0
        self.result = None
        self.arrival_elapsed = 0.0
        self.arrival_reason = 'waiting_for_target'
        self.acquisition_time = None
        self.training = None

    def state(self):
        with self.lock:
            return dict(status=self.status, target_class=self.target_class,
                        generation=self.generation, result=self.result,
                        arrival_elapsed=self.arrival_elapsed, arrival_reason=self.arrival_reason,
                        acquisition_time=self.acquisition_time, training=self.training)

    def update_training(self, episode, total, reward, cells):
        with self.lock:
            self.training = {'episode': int(episode), 'total': int(total),
                             'reward': float(reward), 'cells': int(cells)}
        return self.state()

    def start(self, target_class):
        if target_class != 'red_ball':
            raise ValueError('unsupported target_class')
        with self.lock:
            self.target_class = target_class
            self.status = 'running'
            self.generation += 1
            self.result = None
            self.arrival_elapsed = 0.0
            self.arrival_reason = 'waiting_for_target'
            self.acquisition_time = None
        return self.state()

    def stop(self):
        with self.lock:
            self.status = 'stopped'
        return self.state()

    def finish(self, result, *, generation=None):
        with self.lock:
            if self.status != 'running' or (generation is not None and generation != self.generation):
                return
            self.status = 'finished'
            self.result = result


class DashboardServer:
    def __init__(self, bus, mission, host='127.0.0.1', port=None):
        if port is None:
            port = int(os.environ.get('DIGITAL_FLY_DASHBOARD_PORT','8765'))
        self.bus, self.mission = bus, mission
        self.closed = Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, status, body):
                data = json.dumps(body, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == '/api/state':
                    self.send_json(200, owner.mission.state())
                elif self.path == '/':
                    data = Path(__file__).with_name('static').joinpath('index.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == '/api/telemetry':
                    queue = owner.bus.subscribe(1)
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        self.connection.settimeout(2)
                        while not owner.closed.is_set():
                            try:
                                record = queue.get(timeout=.5)
                                payload = owner.bus.serialize(record)
                                payload['mission'] = owner.mission.state()
                                data = 'data: ' + json.dumps(payload, allow_nan=False) + '\n\n'
                            except Empty:
                                data = ': heartbeat\n\n'
                            self.wfile.write(data.encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                    finally:
                        owner.bus.unsubscribe(queue)
                else:
                    self.send_json(404, {'error': 'not found'})

            def do_POST(self):
                if self.path not in ('/api/mission/start', '/api/mission/stop'):
                    self.send_json(404, {'error': 'not found'})
                    return
                try:
                    self.connection.settimeout(2)
                    length = int(self.headers.get('Content-Length', 0))
                    if not 0 < length <= 4096:
                        raise ValueError('expected small JSON object')
                    body = json.loads(self.rfile.read(length))
                    allowed = {'target_class'} if self.path.endswith('start') else set()
                    if not isinstance(body, dict) or set(body) != allowed:
                        raise ValueError('only mission fields are accepted')
                    result = owner.mission.start(body['target_class']) if allowed else owner.mission.stop()
                except (ValueError, TypeError, KeyError, TimeoutError):
                    self.send_json(400, {'error': 'invalid mission request'})
                    return
                self.send_json(200, result)

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.http.daemon_threads = True
        self.address = self.http.server_address
        self.thread = None

    def start(self):
        self.thread = Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.closed.set()
        if self.thread is not None:
            self.http.shutdown()
            self.thread.join(timeout=3)
        self.http.server_close()
