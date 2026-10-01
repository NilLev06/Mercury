import http.client
import json
import unittest
from fly_brain_rc.telemetry import TelemetryBus
from fly_brain_rc.dashboard.server import DashboardServer, MissionController


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.bus = TelemetryBus()
        self.mission = MissionController()
        self.server = DashboardServer(self.bus, self.mission, port=0)
        self.server.start()

    def tearDown(self):
        self.server.close()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection(*self.server.address, timeout=2)
        conn.request(method, path, None if body is None else json.dumps(body))
        response = conn.getresponse()
        status, data = response.status, response.read()
        conn.close()
        return status, json.loads(data)

    def test_mission_validation_and_stop(self):
        self.assertEqual(self.request('POST', '/api/mission/start', {'target_class': 'cat'})[0], 400)
        self.assertEqual(self.request('POST', '/api/mission/start', {'target_class': 'red_ball', 'traction': 1})[0], 400)
        self.assertEqual(self.request('POST', '/api/mission/start', {'target_class': 'red_ball'})[1]['status'], 'running')
        self.assertEqual(self.request('POST', '/api/mission/stop', {'steering': 0})[0], 400)
        self.assertEqual(self.request('POST', '/api/mission/stop', {})[1]['status'], 'stopped')
        self.assertEqual(self.request('GET', '/api/state')[1]['status'], 'stopped')

    def test_old_episode_cannot_finish_restarted_mission(self):
        self.mission.start('red_ball')
        generation = self.mission.state()['generation']
        self.mission.start('red_ball')
        self.mission.finish({'stable_arrival': False}, generation=generation)
        self.assertEqual(self.mission.state()['status'], 'running')

    def test_sse_latest_record(self):
        self.bus.publish({'sequence': 42})
        conn = http.client.HTTPConnection(*self.server.address, timeout=2)
        conn.request('GET', '/api/telemetry')
        response = conn.getresponse()
        self.assertEqual(response.getheader('Content-Type'), 'text/event-stream')
        self.assertEqual(json.loads(response.readline().decode()[6:])['sequence'], 42)
        response.close()
        conn.close()

    def test_idle_sse_disconnect_releases_subscription(self):
        import time
        conn = http.client.HTTPConnection(*self.server.address, timeout=2)
        conn.request('GET', '/api/telemetry')
        response = conn.getresponse()
        response.readline()
        response.close()
        conn.close()
        deadline = time.monotonic() + 3
        while self.bus._subscribers and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertEqual(len(self.bus._subscribers), 0)

    def test_shutdown_ends_connected_stream(self):
        import time
        conn = http.client.HTTPConnection(*self.server.address, timeout=2)
        conn.request('GET', '/api/telemetry')
        response = conn.getresponse()
        self.server.close()
        self.assertEqual(response.read().strip(), b': heartbeat')
        deadline = time.monotonic() + 1
        while self.bus._subscribers and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(self.bus._subscribers), 0)
        conn.close()
