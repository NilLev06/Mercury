"""Full-history masked-core learning on broad mostly-neural demonstrations."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM,DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.imitation.curriculum import validation_cases,place_start
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.teacher import BicycleTeacher
from scripts.prepare_neural_rehearsal import NeuralLabelTeacher
from scripts.train_live_core import load_live_source,fit_live_sequences
from scripts.train_precontact_dagger import rotating_collection,collect_precontact,expanded_gate
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write


def label_teacher(env,name,tier,specs,actors,old_names):
    if name in old_names:return NeuralLabelTeacher(actors['old'])
    if specs[name].get('label_teacher')=='planner':return BicycleTeacher(env,cruise=.45,margin=.12)
    return NeuralLabelTeacher(actors[tier])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','complex-teacher','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt'))
    parser.add_argument('--rounds',type=int,default=8);parser.add_argument('--updates',type=int,default=200)
    args=parser.parse_args()
    if args.rounds<1 or args.updates<1:raise ValueError('positive fitting counts required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'run.py').write_text(Path(__file__).read_text());(args.out/'fit.snapshot.py').write_text(Path('scripts/train_live_core.py').read_text())
    hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in [('source',args.source),('replay',args.replay),('baseline',args.baseline),('complex_teacher',args.complex_teacher)]}
    raw=torch.load(args.source,weights_only=True);model=load_live_source(args.source,raw['graph'])
    saved=torch.load(args.replay,weights_only=True)
    if not saved.get('collection_complete') or saved['source_sha256']!=hashes['source']:raise ValueError('incomplete or mismatched replay')
    collection_protocol=json.loads((args.replay.parent/'protocol.json').read_text())
    if saved['baseline_sha256']!=hashes['baseline'] or collection_protocol['complex_actor_sha256']!=hashes['complex_teacher']:
        raise ValueError('demonstration and correction teachers differ')
    if collection_protocol['selection_seeds']!=list(range(1030001,1030009)) or collection_protocol['reserved_final_seeds']!=list(range(1040001,1040009)):
        raise ValueError('collection split protocol differs from broad plan')
    fixed=[{k:v for k,v in e.items() if k!='h'} for e in saved['episodes']]
    anchors=[(e['x'],e['y']) for e in fixed if e['teaching_actor']=='old_source']
    canonical=[e['case'] for e in fixed if e['teaching_actor']=='old_source' and tuple(e['pose'])==(0.,0.,0.)]
    old_names={c.name for c in validation_cases()}
    if set(canonical)!=old_names or len(canonical)!=7:raise ValueError('missing canonical source anchors')
    actors={'old':copy.deepcopy(model)}
    for tier,path in [('simple',args.baseline),('complex',args.complex_teacher)]:
        p=torch.load(path,weights_only=True);actors[tier]=load_checkpoint(path,p['graph'])[0]
    for actor in actors.values():
        for parameter in actor.parameters():parameter.requires_grad_(False)
    development=[v for seed in range(1030001,1030009) for v in mixed_cases(seed,'selection',2)]
    write(args.out/'protocol.json',dict(input_sha256=hashes,selection_seeds=list(range(1030001,1030009)),reserved_final_seeds=list(range(1040001,1040009)),
        rounds=args.rounds,updates=args.updates,learning_rate=1e-5,anchor_weight=4.,opening_weight=1.,full_history=True,
        fit_seeds=[980001+i for i in range(args.rounds)],collection_per_tier=16,recent_correction_rounds=3,
        teacher_selection='old source on canonical old; neural on successful neural training maps; expert only on failed neural training maps',
        gate='old7/priorcomplex6 clean; newcomplex>=58/64 zero contacts on all64; simpleclean>=baseline and contacts<=baseline'))
    baseline_report=evaluate(actors['simple'],development);write(args.out/'baseline-development.json',baseline_report)
    write(args.out/'source-development.json',evaluate(model,development))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);mask=model.mask.clone();recent=[];reports=[];results=[]
    for round_index in range(args.rounds):
        corrections=[]
        for case,pose,tier in rotating_collection(saved['collection_cases'],round_index,16):
            env=None
            try:
                env=FlyCarEnv(SIM,DRIVE,scenario=case,render=True);place_start(env,pose)
                teacher=label_teacher(env,case.name,tier,saved['case_specs'],actors,old_names)
                data,report=collect_precontact(env,model,teacher);data.pop('h')
                data.update(case=case.name,pose=pose,tier=tier,round=round_index,teaching_actor='expert' if isinstance(teacher,BicycleTeacher) else 'neural')
                corrections.append(data);report.update(scenario=case.name,pose=pose,tier=tier,round=round_index,teaching_actor=data['teaching_actor']);reports.append(report)
                print(json.dumps(dict(phase='broad_correction',**report)),flush=True)
            except ValueError as error:reports.append(dict(scenario=case.name,round=round_index,error=str(error)))
            finally:
                if env is not None and env.renderer is not None:env.renderer.close()
        if not corrections:raise ValueError('empty correction round')
        recent=(recent+[corrections])[-3:];replay=fixed+[e for group in recent for e in group]
        torch.save(dict(episodes=replay,source_sha256=hashes['source'],case_specs=saved['case_specs']),args.out/f'replay-{round_index}.pt')
        torch.manual_seed(980001+round_index)
        loss=fit_live_sequences(model,optimizer,replay,anchors,updates=args.updates,anchor_weight=4.,opening_weight=1.)
        assert torch.equal(model.mask,mask) and torch.all(model.effective_weights()[mask==0]==0)
        model.training_metadata['broad_neural_dagger']=dict(input_sha256=hashes,round=round_index,updates=args.updates*(round_index+1),
            learning_rate=1e-5,anchor_weight=4.,full_history=True,cached_h_used=False,teacher_controls_correction_motor=False,
            training_cutoffs=['first_contact','stationary_3seconds','stable_arrival','2400steps'],runtime='standard unmodified autonomous evaluator',
            selection_seeds=list(range(1030001,1030009)),reserved_final_seeds=list(range(1040001,1040009)),topology_verified=True)
        save_checkpoint(args.out/f'candidate-{round_index}.pt',model,optimizer,[],raw['iteration'])
        old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);previous=evaluate(model,mixed_cases(20001,'selection',1))
        expanded=evaluate(model,development) if (round_index+1)%2==0 or round_index+1==args.rounds else None
        gate=expanded is not None and expanded_gate(old,previous,expanded,baseline_report)
        summary=dict(round=round_index,old=old['successes'],old_contacts=old['contacts'],previous_complex=sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in previous['reports']),
            expanded=None if expanded is None else {tier:dict(clean=sum(r['success'] and r['contacts']==0 for r in expanded['reports'] if r['family']==tier),contacts=sum(r['contacts'] for r in expanded['reports'] if r['family']==tier)) for tier in ('simple','complex')},development_gate=gate,loss=loss)
        results.append(summary);write(args.out/f'candidate-{round_index}.json',dict(summary=summary,old=old,previous=previous,expanded=expanded))
        write(args.out/'collection.json',reports);write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True));print(json.dumps(dict(phase='broad_candidate',**summary)),flush=True)
        if gate:break
    assert hashes=={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in [('source',args.source),('replay',args.replay),('baseline',args.baseline),('complex_teacher',args.complex_teacher)]}

if __name__=='__main__':main()
