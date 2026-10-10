"""Frozen-brain DAgger focused on preventing first contact, no runtime rules."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
from fly_brain_rc.imitation.curriculum import validation_cases,place_start
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.observations import encode_observation
from fly_brain_rc.imitation.teacher import BicycleTeacher
from fly_brain_rc.perception.target import MockRedBallDetector,MockBlueObstacleDetector
from scripts.train_neural_ppo import freeze_actor,make_anchors
from scripts.train_mastery_curriculum import mastery_cases
from scripts.train_mixed_curriculum import fit_mixture
from scripts.train_sequential import evaluate
from scripts.improve_navigation import neural_states,write


def collect_precontact(env,model,teacher,*,steps=2400,max_idle=3.,record_poses=False):
    if steps<1 or not math.isfinite(max_idle) or max_idle<=0:raise ValueError('positive collection limits required')
    detectors=(MockRedBallDetector(),MockBlueObstacleDetector());state=None;stop=GoalStop()
    xs=[];ys=[];hs=[];poses=[];action=(0.,0.);idle=stalled=0.;reached=False;cutoff=None
    minimum=body_goal_distance(env)
    for index in range(steps):
        if index%5==0:
            if record_poses:poses.append(env.car_pose())
            sensor=env.sensor_frame();target=detectors[0].detect(sensor,'red_ball');obstacle=detectors[1].detect(sensor,'obstacle')
            x=torch.from_numpy(encode_observation(sensor,target,obstacle,schema=model.observation_schema))
            with torch.no_grad():output,state=model(x[None,None],state)
            action=tuple(output[0,0].tolist())
            xs.append(x);hs.append(state[0].clone());ys.append(output[0,0].clone() if teacher is None else torch.tensor(teacher.action_from_sensor(sensor,target,obstacle) if hasattr(teacher,'action_from_sensor') else teacher.action(),dtype=torch.float32))
        # Teacher supplies labels only: every actual command is neural.
        env.step(*action,observe=False)
        distance=body_goal_distance(env);minimum=min(minimum,distance)
        stationary=abs(env.speed)<.03 and distance>.5
        idle=idle+DRIVE.timestep if stationary else 0.;stalled+=DRIVE.timestep*stationary
        reached=stop.update(distance,env.speed,DRIVE.timestep)
        if reached:break
        if env._contact_active:cutoff='contact';break
        if idle>=max_idle-1e-9:cutoff='stationary';break
    report=dict(success=reached,steps=index+1,contacts=env.collisions,stalled_seconds=stalled,
        min_goal_distance=minimum,final_goal_distance=body_goal_distance(env),final_speed=env.speed,
        collection_cutoff=cutoff or ('arrival' if reached else 'time_limit'),teacher_controls_motor=False)
    data=dict(x=torch.stack(xs),y=torch.stack(ys),h=torch.stack(hs))
    if record_poses:data['poses']=poses
    return data,report



def rotating_collection(pool,round_index,count):
    """Canonical old starts plus a deterministic chunk from each training tier."""
    from scripts.prepare_neural_rehearsal import restore_case
    if count<1 or round_index<0:raise ValueError('positive count and nonnegative round required')
    result=[(c,(0.,0.,0.),'simple') for c in validation_cases()]
    for tier in ('simple','complex'):
        subset=[spec for spec in pool if spec['tier']==tier]
        if not subset:raise ValueError('missing collection tier: '+tier)
        result.extend(restore_case(subset[(round_index*count+i)%len(subset)]) for i in range(min(count,len(subset))))
    return result


def expanded_gate(old,previous,expanded,baseline):
    complex_reports=[r for r in expanded['reports'] if r['family']=='complex']
    simple=[r for r in expanded['reports'] if r['family']=='simple']
    base_simple=[r for r in baseline['reports'] if r['family']=='simple']
    previous_complex=[r for r in previous['reports'] if r['family']=='complex']
    return (old['successes']==7 and old['contacts']==0
        and len(previous_complex)==6 and all(r['success'] and r['contacts']==0 for r in previous_complex)
        and len(complex_reports)>=32 and len(complex_reports)==2*len(simple)
        and sum(r['success'] and r['contacts']==0 for r in complex_reports)>=math.ceil(.9*len(complex_reports))
        and all(r['contacts']==0 for r in complex_reports)
        and len(simple)==len(base_simple)
        and sum(r['success'] and r['contacts']==0 for r in simple)>=sum(r['success'] and r['contacts']==0 for r in base_simple)
        and sum(r['contacts'] for r in simple)<=sum(r['contacts'] for r in base_simple))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=8);parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--expanded-selection',action='store_true')
    parser.add_argument('--resume-optimizer',action='store_true')
    parser.add_argument('--round-offset',type=int,default=0)
    parser.add_argument('--neural-simple-teacher',type=Path)
    parser.add_argument('--neural-complex-teacher',type=Path)
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt'))
    parser.add_argument('--collection-per-tier',type=int,default=8)
    parser.add_argument('--training-probe-count',type=int,default=0)
    args=parser.parse_args()
    if args.rounds<1 or args.updates<1 or args.round_offset<0 or args.collection_per_tier<1 or args.training_probe_count<0:raise ValueError('positive training counts required')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'run.py').write_text(Path(__file__).read_text())
    (args.out/'fit.snapshot.py').write_text(Path('scripts/train_mixed_curriculum.py').read_text())
    torch.set_num_threads(2);raw=torch.load(args.source,weights_only=True);model,loaded_optimizer,_,age=load_checkpoint(args.source,raw['graph']);freeze_actor(model)
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    neural_teachers={}
    from scripts.prepare_neural_rehearsal import NeuralLabelTeacher,core_fingerprint
    for tier,path in [('simple',args.neural_simple_teacher),('complex',args.neural_complex_teacher)]:
        if path:
            teacher_raw=torch.load(path,weights_only=True);neural_teachers[tier]=load_checkpoint(path,teacher_raw['graph'])[0]
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    anchors,reports=make_anchors(model,validation_cases())
    if sum(r['success'] for r in reports)!=7 or sum(r['contacts'] for r in reports):raise ValueError('source must retain old seven without contact')
    torch.save(dict(h=anchors[0],y=anchors[1],reports=reports),args.out/'anchors.pt')
    fixed=[]
    saved=torch.load(args.replay,weights_only=True)
    if saved.get('collection_complete') is False:raise ValueError('incomplete rehearsal collection')
    if saved.get('source_sha256') and saved.get('case_specs') and saved['source_sha256']!=source_hash and saved.get('core_fingerprint')!=core_fingerprint(model):
        raise ValueError('rehearsal source hash does not match frozen core')
    for e in saved['episodes']:
        h,_=neural_states(model,e['x']);fixed.append(dict(e,h=h))
    cases=mastery_cases()
    if saved.get('case_specs'):
        from scripts.prepare_neural_rehearsal import restore_case
        cases.update({name:restore_case(spec) for name,spec in saved['case_specs'].items()})
    complex_names=list(dict.fromkeys(e['case'] for e in fixed if e['tier']=='complex'))
    collection_cases=[(c,(0.,0.,0.),'simple') for c in validation_cases()]+[cases[n] for n in complex_names]
    probe_names=complex_names
    if args.training_probe_count and len(complex_names)>args.training_probe_count:
        probe_names=[complex_names[int(i*len(complex_names)/args.training_probe_count)] for i in range(args.training_probe_count)]
    expanded_cases=[v for seed in range(930001,930005) for v in mixed_cases(seed,'selection',2)] if args.expanded_selection else None
    baseline_expanded=None
    if expanded_cases:
        base_raw=torch.load(args.baseline,weights_only=True);baseline_model,*_=load_checkpoint(args.baseline,base_raw['graph'])
        baseline_expanded=evaluate(baseline_model,expanded_cases)
        write(args.out/'baseline-expanded-selection.json',baseline_expanded)
        write(args.out/'source-expanded-selection.json',evaluate(model,expanded_cases))
    write(args.out/'protocol.json',dict(source_sha256=source_hash,replay=str(args.replay.resolve()),
        replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),baseline=str(args.baseline.resolve()) if expanded_cases else None,
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest() if expanded_cases else None,
        expanded_selection_seeds=list(range(930001,930005)) if expanded_cases else None,
        reserved_final_seeds=list(range(950001,950005)) if expanded_cases else None,
        neural_teachers={tier:dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for tier,path in [('simple',args.neural_simple_teacher),('complex',args.neural_complex_teacher)] if path},
        resume_optimizer=args.resume_optimizer,round_offset=args.round_offset,core_fingerprint=core_fingerprint(model),
        collection_per_tier=args.collection_per_tier,training_probe_cases=probe_names,updates=args.updates,rounds=args.rounds,
        success_gate='old7clean, earlier complex6clean, new complex>=29/32 with no contacts, new simple>=baseline clean successes and <=contacts'))
    optimizer=loaded_optimizer if args.resume_optimizer else torch.optim.Adam(model.parameters(),lr=3e-4)
    if args.resume_optimizer and not optimizer.state:raise ValueError('source optimizer has no state to resume')
    recent=[];collection=[];results=[]
    for local_index in range(args.rounds):
        round_index=local_index+args.round_offset
        torch.manual_seed(120000+round_index);corrections=[]
        if saved.get('collection_cases'):collection_cases=rotating_collection(saved['collection_cases'],round_index,args.collection_per_tier)
        for case,pose,tier in collection_cases:
            env=None
            try:
                env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
                teacher=NeuralLabelTeacher(neural_teachers[tier]) if tier in neural_teachers else BicycleTeacher(env,cruise=.45,margin=.12)
                data,report=collect_precontact(env,model,teacher)
                data.update(case=case.name,pose=pose,tier=tier,beta=0.,round=round_index);corrections.append(data)
                report.update(scenario=case.name,tier=tier,round=round_index);collection.append(report)
                print(json.dumps(dict(phase='precontact_collection',**report)),flush=True)
            except ValueError as error:
                collection.append(dict(scenario=case.name,round=round_index,error=str(error)))
            finally:
                if env is not None and env.renderer is not None:env.renderer.close()
        if not corrections:raise ValueError('no correction histories collected')
        recent=(recent+[corrections])[-3:];replay=fixed+[e for group in recent for e in group]
        torch.save(dict(episodes=replay,source_sha256=source_hash,case_specs=saved.get('case_specs',{}),
            collection_cases=saved.get('collection_cases',[])),args.out/f'replay-{round_index}.pt')
        torch.manual_seed(130000+round_index)
        losses=fit_mixture(model,optimizer,replay,anchors,complex_fraction=.5,updates=args.updates)
        assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
        model.training_metadata.pop('fit_scope',None)
        model.training_metadata['precontact_dagger']=dict(source=str(args.source.resolve()),source_sha256=source_hash,round=round_index,
            updates=args.updates*(round_index+1),fit_scope='readout.residual only',core_frozen_verified=True,
            teacher={tier:('neural' if tier in neural_teachers else 'planner') for tier in ('simple','complex')},teacher_controls_motor=False,training_cutoffs=['first_contact','stationary_3seconds','stable_arrival','2400steps'],
            runtime_cutoffs='unchanged standard evaluator',learning_rate=3e-4,old_source_anchor_weight=1.,complex_fraction=.5,
            replay=str((args.out/f'replay-{round_index}.pt').resolve()),collection_seed=120000+round_index,fit_seed=130000+round_index)
        path=args.out/f'candidate-{round_index}.pt';save_checkpoint(path,model,optimizer,[],age)
        old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);new=evaluate(model,mixed_cases(20001,'selection',1))
        complex_reports=[r for r in new['reports'] if r['family']=='complex']
        gate=old['successes']==7 and old['contacts']==0 and sum(r['success'] for r in complex_reports)>=math.ceil(.9*len(complex_reports)) and all(r['contacts']==0 for r in complex_reports)
        expanded=evaluate(model,expanded_cases) if expanded_cases and ((round_index+1)%2==0 or gate or local_index+1==args.rounds) else None
        if expanded_cases:gate=expanded is not None and expanded_gate(old,new,expanded,baseline_expanded)
        training=evaluate(model,[cases[n] for n in probe_names]) if (round_index+1)%2==0 or gate or local_index+1==args.rounds else None
        summary=dict(round=round_index,old=old['successes'],old_contacts=old['contacts'],new=new['successes'],new_contacts=new['contacts'],
            complex=sum(r['success'] for r in complex_reports),complex_contacts=sum(r['contacts'] for r in complex_reports),development_gate=gate,
            training=None if training is None else training['successes'],training_episodes=len(probe_names),expanded=None if expanded is None else dict(
                complex_clean=sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in expanded['reports']),
                complex_contacts=sum(r['contacts'] for r in expanded['reports'] if r['family']=='complex'),
                simple_clean=sum(r['family']=='simple' and r['success'] and r['contacts']==0 for r in expanded['reports']),
                simple_contacts=sum(r['contacts'] for r in expanded['reports'] if r['family']=='simple')))
        results.append(summary);write(args.out/f'candidate-{round_index}-development.json',dict(old=old,new=new,expanded=expanded,training=training,losses=losses,summary=summary))
        write(args.out/'collection.json',collection);write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True))
        print(json.dumps(dict(phase='precontact_candidate',**summary)),flush=True)
        if gate:break
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash


if __name__=='__main__':main()
