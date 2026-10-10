"""Record successful neural teachers with the student's actual sensor schema.

Scene metadata is only for reconstructing training environments and scoring.
It is never concatenated into policy inputs. No live checkpoint writes.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import Scenario
from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
from fly_brain_rc.imitation.curriculum import validation_cases,place_start
from fly_brain_rc.imitation.model import load_checkpoint,graph_signature
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.observations import encode_observation
from fly_brain_rc.imitation.teacher import BicycleTeacher
from fly_brain_rc.perception.target import MockRedBallDetector,MockBlueObstacleDetector
from scripts.train_mastery_curriculum import mastery_cases
from scripts.train_sequential import episode
from scripts.improve_navigation import neural_states,write


def core_fingerprint(model):
    digest=hashlib.sha256((model.observation_schema+graph_signature(model.graph)).encode())
    for name,tensor in sorted(model.state_dict().items()):
        if name.startswith('readout.'):continue
        value=tensor.detach().cpu().contiguous()
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def case_spec(case,pose,tier):
    return dict(name=case.name,obstacles=[dict(position=list(p),colour=list(c)) for p,c in case.obstacles],
                goal=list(case.goal),pose=list(pose),tier=tier)


def restore_case(spec):
    return (Scenario(spec['name'],tuple((tuple(o['position']),tuple(o['colour'])) for o in spec['obstacles']),
                     tuple(spec['goal'])),tuple(spec['pose']),spec['tier'])


def collect_neural_teacher(env,actor,student,*,steps=2400):
    if type(steps) is not int or steps<1:raise ValueError('positive integer steps required')
    detectors=(MockRedBallDetector(),MockBlueObstacleDetector());state=None;stop=GoalStop()
    xs=[];ys=[];action=(0.,0.);stalled=0.;reached=False;minimum=body_goal_distance(env)
    actor.eval();student.eval()
    for index in range(steps):
        if index%5==0:
            sensor=env.sensor_frame();target=detectors[0].detect(sensor,'red_ball');obstacle=detectors[1].detect(sensor,'obstacle')
            x=torch.from_numpy(encode_observation(sensor,target,obstacle,schema=student.observation_schema))
            actor_x=torch.from_numpy(encode_observation(sensor,target,obstacle,schema=actor.observation_schema))
            with torch.no_grad():output,state=actor(actor_x[None,None],state)
            action=tuple(output[0,0].tolist());xs.append(x);ys.append(output[0,0].clone())
        env.step(*action,observe=False)
        distance=body_goal_distance(env);minimum=min(minimum,distance)
        stalled+=DRIVE.timestep*(abs(env.speed)<.03 and distance>.5)
        reached=stop.update(distance,env.speed,DRIVE.timestep)
        if reached:break
    x=torch.stack(xs);h,_=neural_states(student,x)
    return dict(x=x,y=torch.stack(ys),h=h),dict(success=reached,steps=index+1,contacts=env.collisions,
        stalled_seconds=stalled,min_goal_distance=minimum,final_goal_distance=body_goal_distance(env),final_speed=env.speed,
        stable_stop_seconds=stop.elapsed,arrival_criterion='body-centre-0.5m-speed0.03-hold0.5',teaching_actor_controls_motor=True)


class NeuralLabelTeacher:
    """Training-only labels from a recurrent actor on the student's sensor frame."""
    def __init__(self,actor):self.actor=actor;self.state=None;actor.eval()
    def action_from_sensor(self,sensor,target,obstacle):
        x=torch.from_numpy(encode_observation(sensor,target,obstacle,schema=self.actor.observation_schema))
        with torch.no_grad():output,self.state=self.actor(x[None,None],self.state)
        return tuple(output[0,0].tolist())


def demonstration_attempts(primary,fallback=None):
    def attempt(collector):
        try:return collector()
        except ValueError as error:return None,dict(success=False,contacts=0,error=str(error))
    attempts=[attempt(primary)]
    if fallback is not None and not (attempts[0][1]['success'] and attempts[0][1]['contacts']==0):
        attempts.append(attempt(fallback))
    return attempts


def neural_episode(actor,student,case,pose):
    env=None
    try:
        env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
        return collect_neural_teacher(env,actor,student)
    finally:
        if env is not None and env.renderer is not None:env.renderer.close()


def expert_episode(student,case,pose):
    x,y,report=episode(student,case,pose,beta=1.,teacher_factory=lambda env:BicycleTeacher(env,cruise=.45,margin=.12))
    h,_=neural_states(student,x)
    return dict(x=x,y=y,h=h),report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('artifacts/imitation/experiments/frozen-navigation-final-20261009/selected.pt'))
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--complex-actor',type=Path)
    parser.add_argument('--fallback-expert',action='store_true')
    for name,default in [('simple-seed-start',810001),('complex-seed-start',820001),('seed-count',10),
                         ('selection-seed-start',930001),('selection-seed-count',4),('final-seed-start',950001),('final-seed-count',4)]:
        parser.add_argument('--'+name,type=int,default=default)
    args=parser.parse_args()
    if min(args.seed_count,args.selection_seed_count,args.final_seed_count)<1:raise ValueError('positive seed counts required')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'run.py').write_text(Path(__file__).read_text());torch.set_num_threads(2)
    hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in [('source',args.source),('baseline',args.baseline)]}
    raw=torch.load(args.source,weights_only=True);student,*_=load_checkpoint(args.source,raw['graph'])
    raw=torch.load(args.baseline,weights_only=True);baseline,*_=load_checkpoint(args.baseline,raw['graph'])
    complex_actor=student
    if args.complex_actor:
        hashes['complex_actor']=hashlib.sha256(args.complex_actor.read_bytes()).hexdigest()
        raw=torch.load(args.complex_actor,weights_only=True);complex_actor,*_=load_checkpoint(args.complex_actor,raw['graph'])
    jobs=[];registry={};pool=[];episodes=[];reports=[];anchor_h=[];anchor_y=[];completed_jobs=0
    for c in validation_cases():
        registry[c.name]=case_spec(c,(0.,0.,0.),'simple')
        for p in [(0.,0.,0.),(0.,.025,.035),(0.,-.025,-.035)]:jobs.append((c,p,'simple','old_source'))
    for seed in range(args.simple_seed_start,args.simple_seed_start+args.seed_count):
        for c,p,t in mixed_cases(seed,'train',0):
            if t=='simple':jobs.append((c,p,t,'baseline'));registry[c.name]=case_spec(c,p,t);pool.append(registry[c.name])
    for c,p,t in mastery_cases().values():
        if t=='complex':jobs.append((c,p,t,'source_complex'));registry[c.name]=case_spec(c,p,t)
    for seed in range(args.complex_seed_start,args.complex_seed_start+args.seed_count):
        for c,p,t in mixed_cases(seed,'train',2):
            if t=='complex':jobs.append((c,p,t,'new_source_complex' if args.complex_actor else 'gentle_teacher'));registry[c.name]=case_spec(c,p,t);pool.append(registry[c.name])
    write(args.out/'protocol.json',dict(source_sha256=hashes['source'],baseline_sha256=hashes['baseline'],
        training_simple_seeds=list(range(args.simple_seed_start,args.simple_seed_start+args.seed_count)),training_complex_seeds=list(range(args.complex_seed_start,args.complex_seed_start+args.seed_count)),
        selection_seeds=list(range(args.selection_seed_start,args.selection_seed_start+args.selection_seed_count)),reserved_final_seeds=list(range(args.final_seed_start,args.final_seed_start+args.final_seed_count)),
        complex_actor_sha256=hashes.get('complex_actor'),fallback_expert=args.fallback_expert,accepted_replay='stable success and zero contacts only',cached_states='qualified student core',jobs=len(jobs)))
    def save():
        torch.save(dict(episodes=episodes,anchors=dict(h=torch.cat(anchor_h),y=torch.cat(anchor_y)) if anchor_h else None,
            source_sha256=hashes['source'],core_fingerprint=core_fingerprint(student),baseline_sha256=hashes['baseline'],case_specs=registry,collection_cases=pool,
            collection=reports,collection_complete=completed_jobs==len(jobs)),args.out/'rehearsal.pt')
        write(args.out/'collection.json',reports)
        write(args.out/'result.json',dict(attempted=len(reports),accepted=len(episodes),planned=len(jobs),completed_jobs=completed_jobs,
            by_group={g:dict(attempted=sum(r['group']==g for r in reports),clean=sum(r.get('accepted',False) and r['group']==g for r in reports))
                      for g in dict.fromkeys(r['group'] for r in reports)},complete=completed_jobs==len(jobs)))
    for i,(c,p,t,g) in enumerate(jobs):
        actor=baseline if g=='baseline' else complex_actor if g in ('source_complex','new_source_complex') else student
        primary=(lambda:expert_episode(student,c,p)) if g=='gentle_teacher' else (lambda:neural_episode(actor,student,c,p))
        fallback=(lambda:expert_episode(student,c,p)) if args.fallback_expert and g not in ('old_source','gentle_teacher') else None
        attempts=demonstration_attempts(primary,fallback)
        registry[c.name]['label_teacher']='planner' if len(attempts)>1 or g=='gentle_teacher' else 'neural'
        for attempt_index,(data,r) in enumerate(attempts):
            group=g if attempt_index==0 else g+'_expert';accepted=r['success'] and r['contacts']==0
            r.update(scenario=c.name,pose=p,tier=t,group=group,accepted=accepted,job_index=i);reports.append(r)
            if accepted:
                data.update(case=c.name,pose=p,tier=t,teaching_actor=group,layout=case_spec(c,p,t));episodes.append(data)
                if g=='old_source' and p==(0.,0.,0.):anchor_h.append(data['h']);anchor_y.append(data['y'])
            elif g=='old_source' and p==(0.,0.,0.):raise RuntimeError('qualified source failed canonical old map')
            print(json.dumps(dict(phase='neural_rehearsal',index=i,**r)),flush=True)
        if g=='source_complex' and not (attempts[0][1]['success'] and attempts[0][1]['contacts']==0):pool.append(registry[c.name])
        completed_jobs=i+1
        if completed_jobs%10==0:save()
    save()
    assert len(anchor_h)==7
    for n,p in [('source',args.source),('baseline',args.baseline)]+([('complex_actor',args.complex_actor)] if args.complex_actor else []):
        assert hashes[n]==hashlib.sha256(p.read_bytes()).hexdigest()

if __name__=='__main__':main()
