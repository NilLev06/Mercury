import tempfile
import time
import unittest
from pathlib import Path
import sys


class LabTests(unittest.TestCase):
    def test_imitation_jobs_are_separate_and_baseline_is_strict_boolean(self):
        from fly_brain_rc.dashboard.lab import build_job
        root = Path('/tmp/project')
        args = build_job(root,{'mode':'imitation_train','baseline':False,'episodes':6})
        self.assertIn('train-imitation',args)
        self.assertIn(str(root/'artifacts/imitation/connectome.pt'),args)
        baseline = build_job(root,{'mode':'imitation_train','baseline':True})
        self.assertIn('--baseline',baseline)
        self.assertIn(str(root/'artifacts/imitation/rnn.pt'),baseline)
        self.assertNotIn('bypass-policy.json',' '.join(args))
        with self.assertRaises(ValueError):
            build_job(root,{'mode':'imitation_train','baseline':'true'})
        with self.assertRaises(ValueError):
            build_job(root,{'mode':'imitation_test'})

    def test_restart_waits_for_cleanup_even_when_leader_has_exited(self):
        from fly_brain_rc.dashboard.lab import JobManager
        from threading import Event
        from unittest.mock import patch
        release = Event()
        with tempfile.TemporaryDirectory() as directory, \
             patch('fly_brain_rc.dashboard.lab.build_job',return_value=[sys.executable,'-c','import time; time.sleep(.1)']), \
             patch.object(JobManager,'_stop_process',side_effect=lambda *args:release.wait(2)):
            manager = JobManager(directory)
            try:
                manager.start({'mode':'reflex'})
                manager.stop()
                manager.process.wait(timeout=2)
                manager.thread.join(timeout=.2)
                self.assertEqual(manager.state()['status'],'stopping')
                with self.assertRaises(RuntimeError):
                    manager.start({'mode':'reflex'})
                release.set()
                manager.thread.join(timeout=2)
                self.assertEqual(manager.state()['status'],'stopped')
            finally:
                release.set()
                manager.close()

    def test_http_rejects_cross_origin_and_malformed_request(self):
        from fly_brain_rc.dashboard.lab import JobManager,LabServer
        from threading import Thread
        from http.client import HTTPConnection
        import json
        with tempfile.TemporaryDirectory() as directory:
            server = LabServer(JobManager(directory),port=0)
            thread = Thread(target=server.run,daemon=True)
            thread.start()
            try:
                for body,origin,expected in (({},'https://example.com',403),
                                             ({'mode':'shell'},None,400),
                                             ({'mode':[]},None,400)):
                    connection = HTTPConnection(*server.address,timeout=2)
                    headers = {'Content-Type':'application/json'}
                    if origin:
                        headers['Origin'] = origin
                    connection.request('POST','/api/lab/start',json.dumps(body),headers)
                    response = connection.getresponse()
                    self.assertEqual(response.status,expected)
                    response.read()
                    connection.close()
                self.assertEqual(server.manager.state()['status'],'idle')
            finally:
                server.http.shutdown()
                thread.join(timeout=3)

    def test_stop_kills_descendant_even_after_leader_exits(self):
        from fly_brain_rc.dashboard.lab import JobManager
        from unittest.mock import patch
        child = 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); print("child-ready",flush=True); time.sleep(30)'
        script = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(30)'
        with tempfile.TemporaryDirectory() as directory, \
             patch('fly_brain_rc.dashboard.lab.build_job',return_value=[sys.executable,'-c',script]):
            manager = JobManager(directory)
            try:
                manager.start({'mode':'reflex'})
                deadline = time.monotonic()+3
                while 'child-ready' not in manager.state()['logs'] and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertIn('child-ready',manager.state()['logs'])
                manager.stop()
                deadline = time.monotonic()+12
                while manager.state()['status']=='stopping' and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertEqual(manager.state()['status'],'stopped')
            finally:
                manager.close()

    def test_unknown_modes_fields_and_invalid_counts_rejected(self):
        from fly_brain_rc.dashboard.lab import build_job
        root = Path('/tmp/project')
        for request in ({'mode':'shell'},{'mode':'bypass_train','episodes':0},
                        {'mode':'bypass_train','episodes':True},
                        {'mode':'bypass_train','path':'/tmp/anything'},
                        {'mode':'reflex','scenario':'missing'}):
            with self.assertRaises(ValueError):
                build_job(root,request)
        args = build_job(root,{'mode':'bypass_train','episodes':12})
        self.assertIn('train-bypass',args)
        self.assertIn(str(root/'artifacts/rl/bypass-policy.json'),args)

    def test_serial_jobs_can_stop_finish_and_restart(self):
        from fly_brain_rc.dashboard.lab import JobManager
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            manager = JobManager(Path(directory))
            # Replace only subprocess argv construction, not lifecycle code.
            with patch('fly_brain_rc.dashboard.lab.build_job',return_value=[sys.executable,'-c','import time; print("ready",flush=True); time.sleep(30)']):
                manager.start({'mode':'reflex'})
                with self.assertRaises(RuntimeError):
                    manager.start({'mode':'reflex'})
                manager.stop()
                deadline = time.monotonic()+5
                while manager.state()['status'] in ('running','stopping') and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertEqual(manager.state()['status'],'stopped')
            with patch('fly_brain_rc.dashboard.lab.build_job',return_value=[sys.executable,'-c','print("done")']):
                manager.start({'mode':'reflex'})
                deadline = time.monotonic()+5
                while manager.state()['status']=='running' and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertEqual(manager.state()['status'],'finished')
                self.assertIn('done','\n'.join(manager.state()['logs']))
            manager.close()
