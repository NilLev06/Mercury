"""Read-only catalogue and bounded worker lifecycle for neural demonstrations."""
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock,Thread
import uuid
from ..imitation.curriculum import validation_cases
from ..imitation.navigation_cases import mixed_cases

FAVOURITE_SHA='e7155c9a2104a61f2936989a7d0b5b8bd2902e2e09ed2f1925c7c018958b5702'
FAVOURITE_PATH='artifacts/imitation/versions/main-23of32.pt'
ACTIVE=('starting','running','stopping')


def catalogue(root):
    root=Path(root);known={}
    independent=root/'artifacts/imitation/experiments/independent-paired-navigation-20261010/source.json'
    previous=root/'artifacts/imitation/experiments/frozen-navigation-final2-20261009/result.json'
    for path in (independent,previous):
        if not path.exists():continue
        try:
            data=json.loads(path.read_text())
            rows=(data['reports']+data['legacy']['old']['reports']) if path==independent else data['reports']['candidate']['reports']
            for row in rows:known[row['scenario']]=row
        except (ValueError,KeyError,TypeError):continue
    cases=[(c,(0.,0.,0.),'simple','old') for c in validation_cases()]
    cases += [(c,p,t,'complex32') for s in range(950001,950005) for c,p,t in mixed_cases(s,'final',2) if t=='complex']
    cases += [(c,p,t,'independent') for s in range(1050001,1050017) for c,p,t in mixed_cases(s,'final',2)]
    old_names={'corridor':'Прямой коридор','turn_corridor':'Поворот коридора','slalom':'Слалом','blocked_goal':'Закрытая цель','single_cylinder':'Один цилиндр'}
    maps=[];counts={}
    for case,pose,tier,group in cases:
        counts[group]=counts.get(group,0)+1
        title=old_names.get(case.name,case.name.replace('_',' ').capitalize()) if group=='old' else f"{'Сложная' if tier=='complex' else 'Простая'} карта · {counts[group]:03d}"
        maps.append(dict(id=case.name,name=case.name,title=title,group=group,tier=tier,
            obstacles=[dict(position=list(p),colour=list(c)) for p,c in case.obstacles],goal=list(case.goal),pose=list(pose),known_result=known.get(case.name)))
    return dict(maps=maps,model=dict(label='Фаворит 23/32',sha256=FAVOURITE_SHA,neurons=255,edges=4088),
                arena=dict(half=3.,obstacle_radius=.28,wheelbase=.18,track=.36))


def validate_request(request,maps):
    if not isinstance(request,dict) or set(request)-{'map_id','speed'}:raise ValueError('Допустимы только карта и скорость просмотра.')
    key=request.get('map_id');speed=request.get('speed',1)
    if not isinstance(key,str) or key not in maps:raise ValueError('Неизвестная карта.')
    if type(speed) not in (int,float) or speed not in (.5,1,2,4):raise ValueError('Скорость просмотра: 0.5, 1, 2 или 4.')
    return key,speed


class GalleryManager:
    def __init__(self,root):
        self.root=Path(root).resolve();self.catalogue=catalogue(self.root)
        self.maps={m['id']:m for m in self.catalogue['maps']};self.lock=Lock()
        self.process=self.thread=None;self.cancel_file=None;self.status='idle';self.run_id=None;self.map_id=None
        self.speed=1;self.frame=None;self.result=None;self.error=None;self.trail=deque(maxlen=500)
        self.history={};self.logs=deque(maxlen=5)

    def _command(self,key,speed,cancel_file):
        path=self.root/FAVOURITE_PATH
        if not path.exists():raise ValueError('Не найдена сохранённая модель фаворита.')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FAVOURITE_SHA:raise ValueError('Контрольная сумма фаворита изменилась. Запуск остановлен.')
        python=Path(sys.executable);mjpython=python.with_name('mjpython')
        executable=mjpython if sys.platform=='darwin' and mjpython.exists() else python
        return [str(executable),str(self.root/'scripts/run_gallery_episode.py'),'--map-id',key,'--speed',str(speed),'--cancel-file',str(cancel_file)]

    def start(self,request):
        key,speed=validate_request(request,self.maps)
        with self.lock:
            if self.status in ACTIVE:raise RuntimeError('Сначала останови текущую поездку.')
            directory=self.root/'.local/gallery';directory.mkdir(parents=True,exist_ok=True)
            run_id=uuid.uuid4().hex;cancel=directory/(run_id+'.cancel')
            command=self._command(key,speed,cancel)
            process=subprocess.Popen(command,cwd=self.root,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',start_new_session=True)
            self.process=process;self.cancel_file=cancel;self.run_id=run_id;self.map_id=key;self.speed=speed
            self.frame=self.result=self.error=None;self.logs.clear();self.trail.clear();self.status='starting'
            self.thread=Thread(target=self._watch,args=(process,run_id,cancel),daemon=True);self.thread.start()
            return self._snapshot()

    def _receive(self,run_id,record):
        with self.lock:
            if run_id!=self.run_id:return
            kind=record.get('type')
            if kind=='frame':
                self.frame=record['frame'];self.trail.append(list(self.frame['pose'][:2]))
                if self.status=='starting':self.status='running'
            elif kind=='result':self.result=record['result']
            elif kind=='error':self.error=str(record.get('error','Ошибка симулятора'))

    def _watch(self,process,run_id,cancel):
        try:
            for line in process.stdout:
                try:
                    record=json.loads(line)
                    if isinstance(record,dict):self._receive(run_id,record)
                except (ValueError,KeyError,TypeError):
                    with self.lock:
                        if run_id==self.run_id:self.logs.append(line.strip()[:500])
            code=process.wait()
            with self.lock:
                if run_id!=self.run_id:return
                cancelled=self.status=='stopping' or bool(self.result and self.result.get('cancelled'))
                if cancelled:self.status='stopped'
                elif code!=0 or self.result is None or self.error:
                    self.status='error';self.error=self.error or '\n'.join(self.logs) or f'Симулятор завершился с кодом {code}.'
                else:self.status='finished'
                if self.status=='finished':self.history[self.map_id]=dict(self.result)
        finally:
            process.stdout.close();cancel.unlink(missing_ok=True)

    def stop(self):
        with self.lock:
            if self.status not in ACTIVE or self.status=='stopping':return self._snapshot()
            self.status='stopping';self.cancel_file.touch();process=self.process
            Thread(target=self._stop_process,args=(process,),daemon=True).start()
            return self._snapshot()

    @staticmethod
    def _stop_process(process):
        try:process.wait(timeout=2);return
        except subprocess.TimeoutExpired:pass
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGKILL):
            try:os.killpg(process.pid,sig)
            except ProcessLookupError:return
            try:process.wait(timeout=1);return
            except subprocess.TimeoutExpired:pass

    def _snapshot(self):
        return copy.deepcopy(dict(status=self.status,run_id=self.run_id,map_id=self.map_id,speed=self.speed,frame=self.frame,
            trail=list(self.trail),result=self.result,error=self.error,history=self.history))

    def state(self):
        with self.lock:return self._snapshot()

    def close(self):
        self.stop()
        if self.thread:self.thread.join(timeout=6)
