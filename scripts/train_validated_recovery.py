"""Executed, clean training recoveries; matched frozen/live favourite experiment."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
from fly_brain_rc.imitation.curriculum import validation_cases,place_start
from fly_brain_rc.imitation.model import save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.observations import encode_observation
from fly_brain_rc.imitation.teacher import BicycleTeacher
from fly_brain_rc.perception.target import MockRedBallDetector,MockBlueObstacleDetector
from scripts.audit_neural_recovery import replay_prefix,load
from scripts.improve_navigation import write
from scripts.train_live_core import load_live_source,fit_live_sequences
from scripts.train_neural_ppo import freeze_actor
from scripts.train_precontact_dagger import collect_precontact,rotating_collection,expanded_gate
from scripts.train_sequential import evaluate


def route_points(poses,commands):
    """Use actual movement direction: reverse throttle can still be braking."""
    if len(poses)!=len(commands)+1:raise ValueError('one endpoint per command required')
    route=[]
    for before,after,command in zip(poses,poses[1:],commands):
        dx,dy=after[0]-before[0],after[1]-before[1]
        if math.hypot(dx,dy)<1e-5:continue
        direction=1 if dx*math.cos(before[2])+dy*math.sin(before[2])>=0 else -1
        route.append((*after,direction,float(command[1])))
    return route


def join_recovery(prefix,suffix,start):
    if not 0<=start<=len(prefix['x']) or not len(suffix['x']):raise ValueError('invalid recovery window')
    if len(prefix['x'])!=len(prefix['y']) or len(suffix['x'])!=len(suffix['y']):raise ValueError('unaligned labels')
    return dict(x=torch.cat((prefix['x'][:start],suffix['x'])),
                y=torch.cat((prefix['y'][:start],suffix['y'])),supervision_start=start)


class ReferenceRouteTeacher(BicycleTeacher):
    def __init__(self,env,route):
        # Initialise without running A* when a demonstrated route is nearby.
        self.env=env;self.cruise=.45;self.margin=.12
        self.path=list(route);self.cursor=0;self.planned_pose=env.car_pose();self.escape=None
        x,y,yaw=env.car_pose()
        if route:
            self.cursor=min(range(len(route)),key=lambda i:
                math.hypot(route[i][0]-x,route[i][1]-y)+.03*abs((route[i][2]-yaw+math.pi)%(2*math.pi)-math.pi))
        else:self.path=self._plan()


def reference_route(case,pose,demo,steps):
    env=FlyCarEnv(SIM,DRIVE,scenario=case,render=False);place_start(env,pose)
    poses=[env.car_pose()]
    if (steps+4)//5!=len(demo['y']):raise ValueError('demonstration steps differ')
    for index,command in enumerate(demo['y']):
        for _ in range(min(5,steps-index*5)):env.step(*command.tolist(),observe=False)
        poses.append(env.car_pose())
    if env.collisions or body_goal_distance(env)>.5:raise ValueError('demonstrated route failed reconstruction')
    return route_points(poses,demo['y'])


def teacher_suffix(env,teacher,schema,steps):
    detectors=(MockRedBallDetector(),MockBlueObstacleDetector());xs=[];ys=[];stop=GoalStop();success=False
    for i in range(steps):
        if i%5==0:
            sensor=env.sensor_frame();target=detectors[0].detect(sensor,'red_ball');obstacle=detectors[1].detect(sensor,'obstacle')
            xs.append(torch.from_numpy(encode_observation(sensor,target,obstacle,schema=schema)))
            action=teacher.action();ys.append(torch.tensor(action,dtype=torch.float32))
        env.step(*action,observe=False)
        success=stop.update(body_goal_distance(env),env.speed,DRIVE.timestep)
        if env.collisions or success:break
    return dict(x=torch.stack(xs),y=torch.stack(ys)),dict(success=success,contacts=env.collisions,steps=i+1,
        final_goal_distance=body_goal_distance(env),final_speed=env.speed)


def collect_recovery(model,case,pose,route):
    env=None
    try:
        env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
        prefix,report=collect_precontact(env,model,None,record_poses=True)
    finally:
        if env is not None and env.renderer is not None:env.renderer.close()
    if report['success'] and not report['contacts']:
        return None,dict(student=report,accepted=False,reason='already_clean')
    start=max(0,len(prefix['x'])-30);env=None
    try:
        env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
        if start:replay_prefix(env,prefix['y'][:start],start*5)
        if max(abs(a-b) for a,b in zip(env.car_pose(),prefix['poses'][start]))>1e-6:
            raise ValueError('rewound prefix pose did not reproduce')
        if env.collisions:raise ValueError('rewound prefix contacted')
        teacher=ReferenceRouteTeacher(env,route)
        suffix,recovery=teacher_suffix(env,teacher,model.observation_schema,2400-start*5)
        accepted=recovery['success'] and recovery['contacts']==0
        return join_recovery(prefix,suffix,start) if accepted else None,dict(student=report,recovery=recovery,
            prefix_decisions=start,accepted=accepted,reason='clean_recovery' if accepted else 'failed_recovery')
    finally:
        if env is not None and env.renderer is not None:env.renderer.close()


def counts(report):
    return {tier:dict(clean=sum(r['success'] and r['contacts']==0 for r in report['reports'] if r['family']==tier),
        contacts=sum(r['contacts'] for r in report['reports'] if r['family']==tier)) for tier in ('simple','complex')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/imitation/versions/main-23of32.pt'))
    parser.add_argument('--replay',type=Path,default=Path('artifacts/imitation/experiments/broad-neural-replay-20261009/rehearsal.pt'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--collection-per-tier',type=int,default=32)
    parser.add_argument('--stages',type=int,default=4);parser.add_argument('--updates',type=int,default=200)
    args=parser.parse_args()
    if min(args.collection_per_tier,args.stages,args.updates)<1:raise ValueError('positive experiment sizes required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protected=[args.source,Path('artifacts/imitation/connectome.pt'),Path('artifacts/imitation/connectome.best.pt')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    source_hash=hashes[str(args.source)];raw=torch.load(args.source,weights_only=True)
    source=load_live_source(args.source,raw['graph']);saved=torch.load(args.replay,weights_only=True)
    if not saved['collection_complete'] or saved['source_sha256']!=source_hash:raise ValueError('replay source differs')
    fixed=[{k:v for k,v in e.items() if k!='h'} for e in saved['episodes']]
    anchors=[(e['x'],e['y']) for e in fixed if e['teaching_actor']=='old_source']
    if len(anchors)!=21:raise ValueError('expected favourite anchors')
    protocol=dict(source_sha256=source_hash,replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        stages=args.stages,updates_per_stage=args.updates,collection_per_tier=args.collection_per_tier,
        recovery_repeats=4,anchor_weight=4.,learning_rates=dict(frozen=1e-4,live=1e-5),
        fit_seeds=[1090001+i for i in range(args.stages)],development_seeds=list(range(1030001,1030009)),
        reserved_final_seeds=list(range(1040001,1040009)),gate='old7/priorcomplex6 clean; complex>=58/64 zero contacts all64; simple>=30/32 clean contacts<=515',
        shared_collection=True,full_history=True,cached_states_used=False,teacher='training only; executed clean recoveries',
        runtime='unchanged neural evaluator',protected_sha256=hashes)
    write(args.out/'protocol.json',protocol)
    (args.out/'run.py').write_text(Path(__file__).read_text())
    (args.out/'fit.snapshot.py').write_text(Path('scripts/train_live_core.py').read_text())
    corrections=[];reports=[]
    for case,pose,tier in rotating_collection(saved['collection_cases'],0,args.collection_per_tier):
        try:
            demo=next(e for e in fixed if e['case']==case.name and tuple(e['pose'])==tuple(pose))
            row=next(r for r in saved['collection'] if r['scenario']==case.name and tuple(r['pose'])==tuple(pose) and r['accepted'])
            route=reference_route(case,pose,demo,row['steps'])
            data,report=collect_recovery(source,case,pose,route)
            if data is not None:
                data.update(case=case.name,pose=pose,tier=tier,teaching_actor='validated_recovery');corrections.append(data)
        except ValueError as error:report=dict(accepted=False,error=str(error))
        report.update(case=case.name,tier=tier);reports.append(report)
        write(args.out/'collection.json',reports);print(json.dumps(dict(phase='validated_collection',**report)),flush=True)
    torch.save(dict(episodes=corrections,source_sha256=source_hash,reports=reports),args.out/'recoveries.pt')
    if not corrections:raise ValueError('no validated recoveries collected')
    replay=fixed+corrections*4
    development=[v for seed in range(1030001,1030009) for v in mixed_cases(seed,'selection',2)]
    baseline=evaluate(load(Path('artifacts/imitation/versions/original-pre-compass.pt')),development)
    source_report=evaluate(source,development)
    write(args.out/'baseline-development.json',baseline);write(args.out/'source-development.json',source_report)
    results=[];mask=source.mask.clone()
    for arm in ('frozen','live'):
        model=copy.deepcopy(source)
        if arm=='frozen':freeze_actor(model);model.trainable_core=True
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.Adam(model.parameters(),lr=protocol['learning_rates'][arm])
        for stage in range(args.stages):
            torch.manual_seed(protocol['fit_seeds'][stage])
            loss=fit_live_sequences(model,optimizer,replay,anchors,updates=args.updates,anchor_weight=4.)
            assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
            assert torch.equal(model.mask,mask) and torch.all(model.effective_weights()[mask==0]==0)
            model.training_metadata['validated_recovery']=dict(protocol,arm=arm,stage=stage,core_frozen_verified=arm=='frozen')
            # Save the actual inference/training scope, not the fitting helper's flag.
            model.trainable_core=arm=='live'
            path=args.out/f'{arm}-{stage}.pt';save_checkpoint(path,model,optimizer,[],raw['iteration'])
            model.trainable_core=True
            old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()])
            previous=evaluate(model,mixed_cases(20001,'selection',1));expanded=evaluate(model,development)
            gate=expanded_gate(old,previous,expanded,baseline)
            result=dict(arm=arm,stage=stage,updates=(stage+1)*args.updates,loss=loss,old_clean=sum(r['success'] and r['contacts']==0 for r in old['reports']),
                old_contacts=old['contacts'],prior_complex_clean=sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in previous['reports']),
                expanded=counts(expanded),development_gate=gate,path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            write(args.out/f'{arm}-{stage}.json',dict(summary=result,old=old,previous=previous,expanded=expanded))
            results.append(result);write(args.out/'result.json',dict(source=counts(source_report),candidates=results,
                accepted_recoveries=len(corrections),collection_cases=len(reports),deployed=False,final_not_used=True))
            print(json.dumps(dict(phase='validated_candidate',**result)),flush=True)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}


if __name__=='__main__':main()
