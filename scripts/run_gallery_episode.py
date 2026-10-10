"""One read-only real simulator episode, streamed to the local map gallery."""
import argparse
import base64
import hashlib
import json
import signal
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.dashboard.gallery import catalogue,FAVOURITE_PATH,FAVOURITE_SHA
from fly_brain_rc.dashboard.server import MissionController
from fly_brain_rc.imitation.curriculum import place_start
from fly_brain_rc.imitation.model import load_checkpoint
from fly_brain_rc.imitation.runner import collect_episode
from fly_brain_rc.imitation.arrival import body_goal_distance
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import Scenario
from fly_brain_rc.perception.target import MockRedBallDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--map-id',required=True)
    parser.add_argument('--speed',type=float,default=1.,choices=(.5,1.,2.,4.));parser.add_argument('--cancel-file',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];path=root/FAVOURITE_PATH
    torch.set_num_threads(2)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FAVOURITE_SHA:raise ValueError('Сохранённая модель изменилась.')
    spec=next((m for m in catalogue(root)['maps'] if m['id']==args.map_id),None)
    if spec is None:raise ValueError('Неизвестная карта.')
    raw=torch.load(path,weights_only=True);model,*_=load_checkpoint(path,raw['graph'])
    case=Scenario(spec['name'],tuple((tuple(o['position']),tuple(o['colour'])) for o in spec['obstacles']),tuple(spec['goal']))
    interrupted=False
    def interrupt(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    def cancelled():return interrupted or args.cancel_file.exists()
    activity=[]
    def neural_hook(module,inputs,output):
        nonlocal activity
        activity=output[1][0].detach().tolist()
    hook=model.register_forward_hook(neural_hook);env=None;last_frame=None
    def publish(record):print(json.dumps(record,allow_nan=False),flush=True)
    try:
        env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,tuple(spec['pose']))
        started=time.monotonic();mission=MissionController(running=True)
        def emit(record):
            nonlocal last_frame
            deadline=started+record.timestamp/args.speed
            while time.monotonic()<deadline and not cancelled():time.sleep(max(0.,min(.02,deadline-time.monotonic())))
            sensor=record.sensor
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(sensor.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise ValueError('Не удалось сформировать изображение камеры.')
            last_frame=dict(step=record.sequence,time=record.timestamp,pose=list(env.car_pose()),speed=env.speed,
                contacts=env.collisions,target_visible=record.target is not None,
                goal_distance=body_goal_distance(env),arrival_elapsed=mission.arrival_elapsed,action=[record.command.traction,record.command.steering],
                ranges=[sensor.tof,sensor.range_left,sensor.range_right,sensor.range_side_left,sensor.range_side_right,sensor.range_rear],
                preview='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode('ascii'),activity=activity)
            publish(dict(type='frame',frame=last_frame))
        _,report=collect_episode(env,model,steps=2400,beta=None,emit=emit,cancel=cancelled,mission=mission)
        if last_frame is not None:
            sensor=env.sensor_frame()
            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(sensor.rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise ValueError('Не удалось сформировать итоговое изображение камеры.')
            last_frame.update(preview='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode('ascii'),
                target_visible=MockRedBallDetector().detect(sensor,'red_ball') is not None,
                ranges=[sensor.tof,sensor.range_left,sensor.range_right,sensor.range_side_left,sensor.range_side_right,sensor.range_rear])
            last_frame.update(step=report['steps'],time=report['steps']*DRIVE.timestep,pose=list(env.car_pose()),speed=env.speed,
                contacts=env.collisions,goal_distance=body_goal_distance(env),arrival_elapsed=report['stable_stop_seconds'])
            publish(dict(type='frame',frame=last_frame))
        report.update(map_id=args.map_id,model_sha256=FAVOURITE_SHA)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=FAVOURITE_SHA:raise ValueError('Контрольная сумма модели изменилась во время просмотра.')
        publish(dict(type='result',result=report))
    finally:
        hook.remove()
        if env is not None and env.renderer is not None:env.renderer.close()


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(type='error',error=str(error)),ensure_ascii=False),flush=True)
        raise
