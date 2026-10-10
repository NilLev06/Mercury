"""Training-only audit: can label actors recover the student's failed states?"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.imitation.curriculum import place_start
from fly_brain_rc.imitation.model import load_checkpoint
from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
from fly_brain_rc.imitation.observations import encode_observation,SCHEMA_SIZES
from fly_brain_rc.imitation.teacher import BicycleTeacher
from fly_brain_rc.perception.target import MockRedBallDetector,MockBlueObstacleDetector
from scripts.prepare_neural_rehearsal import restore_case,NeuralLabelTeacher
from scripts.improve_navigation import write


def load(path):
    payload=torch.load(path,weights_only=True)
    return load_checkpoint(path,payload['graph'])[0]


def replay_prefix(env,commands,steps):
    if steps<1 or (steps+4)//5!=len(commands):raise ValueError('prefix command count mismatch')
    for i in range(steps):env.step(*commands[i//5].tolist(),observe=False)


def continue_teacher(env,teacher,steps=2400):
    detectors=(MockRedBallDetector(),MockBlueObstacleDetector());stop=GoalStop();action=(0.,0.);reached=False
    for i in range(steps):
        if i%5==0:
            if isinstance(teacher,NeuralLabelTeacher):
                sensor=env.sensor_frame();target=detectors[0].detect(sensor,'red_ball');obstacle=detectors[1].detect(sensor,'obstacle')
                action=teacher.action_from_sensor(sensor,target,obstacle)
            else:action=teacher.action()
        env.step(*action,observe=False);reached=stop.update(body_goal_distance(env),env.speed,DRIVE.timestep)
        if reached or env._contact_active:break
    return dict(success=reached,contacts=env.collisions,steps=i+1,final_goal_distance=body_goal_distance(env),final_speed=env.speed)


def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2)
    (out/'run.py').write_text(Path(__file__).read_text())
    folder=Path('artifacts/imitation/experiments/broad-neural-dagger-20261009')
    saved=torch.load('artifacts/imitation/experiments/broad-neural-replay-20261009/rehearsal.pt',weights_only=True)
    replays={}
    reports=json.loads((folder/'collection.json').read_text())
    pool=[r for r in reports if r.get('collection_cutoff')=='stationary' and r['contacts']==0 and r.get('teaching_actor')=='neural' and r['scenario'] in saved['case_specs']]
    eligible=[r for t in ('simple','complex') for r in [x for x in pool if x['tier']==t][:6]];sources={}
    actors={'simple':load('artifacts/imitation/experiments/navigation-20261008/baseline.pt'),
            'complex':load('artifacts/imitation/experiments/mixed-neural-rehearsal-fit-20261009/candidate-5.pt')}
    results=[]
    for report in eligible:
        round_index=report['round']
        if round_index not in replays:replays[round_index]=torch.load(folder/f'replay-{round_index}.pt',weights_only=True)['episodes']
        if round_index not in sources:sources[round_index]=load('artifacts/imitation/experiments/frozen-navigation-final2-20261009/selected.pt' if round_index==0 else folder/f'candidate-{round_index-1}.pt')
        source=sources[round_index]
        e=next(e for e in replays[round_index] if e.get('round')==round_index and e['case']==report['scenario'])
        case,pose,tier=restore_case(saved['case_specs'][e['case']])
        with torch.no_grad():commands=source(e['x'][None])[0][0]
        for kind in ('neural','expert'):
            env=None
            try:
                env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose);replay_prefix(env,commands,report['steps'])
                reproduced=body_goal_distance(env)
                if abs(reproduced-report['final_goal_distance'])>1e-5:raise ValueError('prefix state failed to reproduce')
                if env.collisions:raise ValueError('prefix unexpectedly contacted')
                if kind=='neural':
                    teacher=NeuralLabelTeacher(actors[tier]);size=SCHEMA_SIZES[teacher.actor.observation_schema]
                    with torch.no_grad():_,teacher.state=teacher.actor(e['x'][None,:,:size])
                else:teacher=BicycleTeacher(env,cruise=.45,margin=.12)
                result=continue_teacher(env,teacher)
                row=dict(case=case.name,round=round_index,tier=tier,kind=kind,prefix_steps=report['steps'],prefix_distance=reproduced,**result)
            except ValueError as error:row=dict(case=case.name,tier=tier,kind=kind,error=str(error),success=False,contacts=0)
            finally:
                if env is not None and env.renderer is not None:env.renderer.close()
            results.append(row);print(json.dumps(row),flush=True)
            write(out/'result.json',dict(eligible=len(eligible),eligible_pool=len(pool),reports=results,training_maps_only=True,
                summary={k:dict(episodes=sum(r['kind']==k for r in results),clean=sum(r['kind']==k and r['success'] and r['contacts']==0 for r in results)) for k in ('neural','expert')}))

if __name__=='__main__':main()
