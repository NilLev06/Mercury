"""Persistent loopback-only process launcher. No user-supplied shell commands."""
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from threading import Event, Lock, Thread
from .gallery import GalleryManager
import time
from urllib.request import urlopen
import uuid


MODES = ('imitation_train','imitation_test','imitation_sim',
         'bypass_train','bypass_test','general_train','bypass_sim','general_sim','reflex')
SCENARIOS = ('corridor','turn_corridor','slalom','blocked_goal','random','single_cylinder')


def build_job(root, request):
    if not isinstance(request,dict) or set(request)-{'mode','episodes','steps','seed','scenario','baseline'}:
        raise ValueError('unknown job fields')
    mode = request.get('mode')
    if mode not in MODES:
        raise ValueError('unknown mode')
    baseline = request.get('baseline',False)
    if type(baseline) is not bool:
        raise ValueError('baseline must be boolean')
    values = {}
    for name,default,low,high in (('episodes',120,1,10000),('steps',2400,1,20000),('seed',0,0,2**31-1)):
        value = request.get(name,default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'invalid {name}')
        values[name] = value
    scenario = request.get('scenario','single_cylinder' if mode=='bypass_sim' else 'corridor')
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    command = [sys.executable,str(root/'run.py')]
    if mode.startswith('imitation_'):
        path = root/'artifacts/imitation'/('rnn.pt' if baseline else 'connectome.pt')
        best = path.with_name(path.stem+'.best.pt')
        if mode != 'imitation_train' and best.exists():
            path = best
        if mode != 'imitation_train' and not path.exists():
            raise ValueError('Сначала обучи выбранную imitation-модель.')
        command += ['train-imitation' if mode=='imitation_train' else 'test-imitation',
                    '--episodes',str(1 if mode=='imitation_sim' else values['episodes']),
                    '--steps',str(values['steps']),'--seed',str(values['seed']),
                    '--out',str(path),'--dashboard']
        if baseline:
            command.append('--baseline')
        if mode=='imitation_sim':
            command += ['--viewer','--scenario',scenario]
    elif mode in ('bypass_train','bypass_test','general_train'):
        command += [{'bypass_train':'train-bypass','bypass_test':'test-bypass','general_train':'train-rl'}[mode],
                    '--episodes',str(values['episodes']),'--steps',str(values['steps']),
                    '--seed',str(values['seed']),'--out',str(root/'artifacts/rl'/
                    ('q-policy.json' if mode=='general_train' else 'bypass-policy.json')),'--dashboard']
    else:
        command += ['sim','--scenario',scenario,'--seed',str(values['seed']),'--dashboard']
        if mode != 'reflex':
            path = root/'artifacts/rl'/('bypass-policy.json' if mode=='bypass_sim' else 'q-policy.json')
            if not path.exists():
                raise ValueError('Сначала обучи выбранную модель.')
            command += ['--rl-policy',str(path)]
    return command


class JobManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = Lock()
        self.process = self.thread = None
        self.status = 'idle'
        self.mode = None
        self.logs = deque(maxlen=80)
        self.exit_code = None
        self.worker_port = None
        self.cancel_file = None
        self.started = None
        self.result = None
        self.validation = None
        self.validation_active = False
        self.series = deque(maxlen=300)
        self.stop_done = Event()
        self.stop_done.set()

    def start(self, request):
        command = build_job(self.root,request)
        with self.lock:
            if self.status in ('running','stopping'):
                raise RuntimeError('Сначала останови текущий запуск.')
            with socket.socket() as probe:
                probe.bind(('127.0.0.1',0))
                self.worker_port = probe.getsockname()[1]
            directory = self.root/'.local/lab'
            directory.mkdir(parents=True,exist_ok=True)
            self.cancel_file = directory/(uuid.uuid4().hex+'.cancel')
            env = dict(os.environ,PYTHONUNBUFFERED='1',DIGITAL_FLY_NO_BROWSER='1',
                       DIGITAL_FLY_DASHBOARD_PORT=str(self.worker_port),
                       DIGITAL_FLY_CANCEL_FILE=str(self.cancel_file))
            self.logs.clear()
            self.series.clear()
            self.result = None
            self.validation = None
            self.validation_active = False
            self.exit_code = None
            self.mode = request['mode']
            self.started = time.time()
            self.process = subprocess.Popen(command,cwd=self.root,env=env,
                                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                            text=True,encoding='utf-8',errors='replace',start_new_session=True)
            self.status = 'running'
            self.thread = Thread(target=self._watch,args=(self.process,self.cancel_file),daemon=True)
            self.thread.start()
        return self.state()

    def _watch(self, process, cancel_file):
        try:
            for line in process.stdout:
                with self.lock:
                    self.logs.append(line.rstrip()[:2000])
                    try:
                        report = json.loads(line)
                        if isinstance(report,dict) and 'episode' in report and 'reward' in report:
                            self.series.append(report)
                        elif isinstance(report,dict) and report.get('phase')=='validation_start':
                            self.validation_active = True
                        elif isinstance(report,dict) and report.get('phase')=='validation':
                            self.validation = report['validation']
                            self.validation_active = False
                        elif isinstance(report,dict) and 'successes' in report:
                            self.result = report
                    except (ValueError,TypeError):
                        pass
            code = process.wait()
            with self.lock:
                self.exit_code = code
                self.validation_active = False
                stopping = self.status=='stopping'
                if not stopping:
                    self.status = 'finished' if code==0 else 'error'
            if stopping:
                self.stop_done.wait()
                with self.lock:
                    self.status = 'stopped'
        finally:
            process.stdout.close()
            cancel_file.unlink(missing_ok=True)

    def stop(self):
        with self.lock:
            if self.status not in ('running','stopping'):
                return self._snapshot()
            if self.status == 'stopping':
                return self._snapshot()
            self.status = 'stopping'
            self.stop_done.clear()
            process = self.process
            self.cancel_file.touch()
            training = self.mode in ('bypass_train','bypass_test','general_train',
                                     'imitation_train','imitation_test','imitation_sim')
        # Trainers see the flag on the next tick and checkpoint. Viewer loops
        # have no trainer callback, so receive SIGINT immediately.
        Thread(target=self._stop_job,args=(process,training),daemon=True).start()
        return self.state()

    def _stop_job(self, process, training):
        try:
            self._stop_process(process,training)
        finally:
            self.stop_done.set()

    @staticmethod
    def _stop_process(process, training):
        if training:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pass
        for sig,timeout in ((signal.SIGINT,4),(signal.SIGTERM,3),(signal.SIGKILL,1)):
            try:
                os.killpg(process.pid,sig)
            except ProcessLookupError:
                return
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                process.poll()  # Reap leader, but surviving descendants matter.
                try:
                    os.killpg(process.pid,0)
                except ProcessLookupError:
                    return
                except PermissionError:
                    # Some restricted hosts reject signal-0 probes. Do not
                    # treat that as group disappearance; keep escalation.
                    pass
                time.sleep(.05)

    def _snapshot(self):
        return dict(app='digital-fly-lab',status=self.status,mode=self.mode,
                    started=self.started,exit_code=self.exit_code,logs=list(self.logs),
                    worker_port=self.worker_port,
                    result=self.result,series=list(self.series),validation=self.validation,
                    validation_active=self.validation_active,
                    policies={name:(self.root/'artifacts/rl'/file).exists() for name,file in
                              (('bypass','bypass-policy.json'),('general','q-policy.json'))})

    def state(self):
        with self.lock:
            state = self._snapshot()
        if state['status'] in ('running','stopping') and state['worker_port']:
            try:
                with urlopen(f'http://127.0.0.1:{state["worker_port"]}/api/state',timeout=.25) as response:
                    state['live'] = json.load(response)
            except (OSError,ValueError):
                pass
        return state

    def close(self):
        self.stop()
        if self.thread:
            self.thread.join(timeout=17)


class LabServer:
    def __init__(self, manager, host='127.0.0.1',port=8765):
        if host != '127.0.0.1':
            raise ValueError('lab must bind loopback')
        self.manager = manager
        self.gallery = GalleryManager(manager.root)
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def reply(self,status,body):
                data = json.dumps(body,allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path=='/api/lab/state':
                    self.reply(200,owner.manager.state())
                elif self.path=='/api/gallery/maps':
                    self.reply(200,owner.gallery.catalogue)
                elif self.path=='/api/gallery/state':
                    self.reply(200,owner.gallery.state())
                elif self.path in ('/maps','/maps/','/maps/gallery.css','/maps/gallery.js'):
                    filename={'/maps':'gallery.html','/maps/':'gallery.html','/maps/gallery.css':'gallery.css','/maps/gallery.js':'gallery.js'}[self.path]
                    path=Path(__file__).with_name('static')/filename
                    if not path.exists():self.reply(404,{'error':'gallery asset unavailable'});return
                    data=path.read_bytes();self.send_response(200)
                    self.send_header('Content-Type',{'gallery.html':'text/html; charset=utf-8','gallery.css':'text/css; charset=utf-8','gallery.js':'text/javascript; charset=utf-8'}[filename])
                    self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
                elif self.path=='/':
                    data = Path(__file__).with_name('static').joinpath('lab.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type','text/html; charset=utf-8')
                    self.send_header('Content-Length',str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.reply(404,{'error':'not found'})

            def do_POST(self):
                expected = f'127.0.0.1:{owner.address[1]}'
                if self.headers.get('Host') != expected or self.headers.get('Origin') not in (None,f'http://{expected}'):
                    self.reply(403,{'error':'local same-origin requests only'})
                    return
                if self.path not in ('/api/lab/start','/api/lab/stop','/api/gallery/start','/api/gallery/stop'):
                    self.reply(404,{'error':'not found'})
                    return
                try:
                    self.connection.settimeout(2)
                    length = int(self.headers.get('Content-Length',0))
                    if not 0 < length <= 4096:
                        raise ValueError('expected small JSON object')
                    if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                        raise ValueError('expected application/json')
                    request = json.loads(self.rfile.read(length))
                    if self.path=='/api/gallery/start':
                        result=owner.gallery.start(request)
                    elif self.path=='/api/gallery/stop':
                        if request!={}:raise ValueError('stop accepts no fields')
                        result=owner.gallery.stop()
                    elif self.path.endswith('start'):
                        result = owner.manager.start(request)
                    else:
                        if request != {}:
                            raise ValueError('stop accepts no fields')
                        result = owner.manager.stop()
                except RuntimeError as exc:
                    self.reply(409,{'error':str(exc)})
                    return
                except (ValueError,TypeError,OSError,TimeoutError) as exc:
                    self.reply(400,{'error':str(exc)})
                    return
                self.reply(200,result)
        self.http = ThreadingHTTPServer((host,port),Handler)
        self.http.daemon_threads = True
        self.address = self.http.server_address

    def run(self):
        try:
            self.http.serve_forever()
        finally:
            self.gallery.close()
            self.manager.close()
            self.http.server_close()


def run_lab(root,port=8765,open_path='/'):
    import webbrowser
    url = f'http://127.0.0.1:{port}'
    try:
        with urlopen(url+'/api/lab/state',timeout=.5) as response:
            existing = json.load(response)
        if existing.get('app')=='digital-fly-lab':
            if not os.environ.get('DIGITAL_FLY_NO_BROWSER'):
                webbrowser.open(url+open_path)
            return
    except (OSError,ValueError):
        pass
    server = LabServer(JobManager(root),port=port)
    print(f'Digital Fly Lab: {url} — keep this window open',flush=True)
    if not os.environ.get('DIGITAL_FLY_NO_BROWSER'):
        webbrowser.open(url+open_path)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
