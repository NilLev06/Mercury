"""Expand a coherent teacher curriculum only after autonomous mastery.

A diagnostic successor to simultaneously fitting many conflicting old-source
anchors and new teacher trajectories. No runtime planner or live writes.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.validation import validate
from fly_brain_rc.imitation.teacher import BicycleTeacher
from scripts.train_live_core import load_live_source,fit_live_sequences
from scripts.train_sequential import evaluate,episode
from scripts.improve_navigation import write


def mastery_cases():
    from scripts.collect_mirror_replay import mirror_case,mirror_pose
    cases={c.name:(c,p,t) for seed in range(31000,31004) for c,p,t in mixed_cases(seed,'train',1)}
    cases.update({c.name:(c,(0.,0.,0.),'simple') for c in validation_cases()})
    for c,p,t in list(cases.values()):
        reflected=mirror_case(c)
        cases[reflected.name]=(reflected,mirror_pose(p),t)
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=16)
    parser.add_argument('--updates',type=int,default=500)
    parser.add_argument('--replay',type=Path,default=Path('artifacts/imitation/experiments/sector-navigation-20261009/replay-0.pt'))
    parser.add_argument('--all-old-starts',action='store_true')
    parser.add_argument('--source',type=Path,default=Path('artifacts/imitation/experiments/sector-initial-20261009.pt'))
    parser.add_argument('--correction-beta',type=float)
    parser.add_argument('--teacher-cruise',type=float,default=.45)
    parser.add_argument('--teacher-margin',type=float,default=.12)
    parser.add_argument('--initial-complex',type=int,default=2)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--anchor-weight',type=float,default=.01)
    parser.add_argument('--opening-weight',type=float,default=10.)
    args=parser.parse_args()
    if args.rounds<1 or args.updates<1:raise ValueError('positive training counts required')
    if args.correction_beta is not None and not 0<=args.correction_beta<=1:raise ValueError('invalid correction probability')
    if args.lr<=0 or args.anchor_weight<=0 or args.opening_weight<1:raise ValueError('invalid fitting settings')
    args.out.mkdir(exist_ok=False,parents=True)
    (args.out/'run.py').write_text(Path(__file__).read_text())
    (args.out/'fit.snapshot.py').write_text(Path('scripts/train_live_core.py').read_text())
    torch.set_num_threads(2)
    source=args.source
    payload=torch.load(source,weights_only=True)
    graph=torch.load('artifacts/imitation/experiments/navigation-20261008/baseline.pt',weights_only=True)['graph']
    model=load_live_source(source,graph)
    saved=torch.load(args.replay,weights_only=True)
    old=[]
    for case in validation_cases():
        # First occurrence is the exact canonical start, subsequent entries
        # are nearby start perturbations; begin with nine coherent routes.
        names=(case.name,case.name+'-physical-mirror') if args.all_old_starts else (case.name,)
        matches=[e for e in saved['episodes'] if e['case'] in names]
        if not matches:raise ValueError('missing old teacher route: '+case.name)
        old.extend(matches if args.all_old_starts else matches[:1])
    complex_pool=[e for e in saved['episodes'] if e['tier']=='complex']
    cases=mastery_cases()
    if not 1<=args.initial_complex<=len(complex_pool):raise ValueError('initial complex count outside replay')
    if any(e['case'] not in cases for e in complex_pool):raise ValueError('unresolved complex route in replay')
    anchors=[(e['x'],e['y']) for e in old]
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    count=args.initial_complex;results=[];mask=model.mask.clone();correction_rounds=[];collection=[]
    for round_index in range(args.rounds):
        if args.correction_beta is not None:
            torch.manual_seed(91000+round_index)
            corrections=[]
            collection_cases=[(c,(0.,0.,0.),'simple') for c in validation_cases()]
            collection_cases += [cases[e['case']] for e in complex_pool[:count]]
            for case,pose,tier in collection_cases:
                try:
                    x,y,r=episode(model,case,pose,beta=args.correction_beta,
                        teacher_factory=lambda env:BicycleTeacher(env,cruise=args.teacher_cruise,margin=args.teacher_margin))
                    corrections.append(dict(x=x,y=y,tier=tier,case=case.name,beta=args.correction_beta))
                    collection.append(dict(r,round=round_index,tier=tier))
                except ValueError as error:
                    collection.append(dict(round=round_index,case=case.name,error=str(error)))
            correction_rounds=(correction_rounds+[corrections])[-2:]
        replay=old+complex_pool[:count]+[e for group in correction_rounds for e in group]
        cache=args.out/f'replay-{round_index}.pt'
        torch.save(dict(episodes=replay,anchors=anchors,collection=collection),cache)
        torch.manual_seed(81000+round_index)
        loss=fit_live_sequences(model,optimizer,replay,anchors,updates=args.updates,anchor_weight=args.anchor_weight,opening_weight=args.opening_weight)
        assert torch.equal(mask,model.mask)
        errors=[]
        with torch.no_grad():
            for e in replay:
                prediction=model(e['x'][None])[0][0]
                errors.append(dict(case=e['case'],mse=float((prediction-e['y']).square().mean()),max_error=float((prediction-e['y']).abs().max())))
        model.training_metadata['fit_scope']='all'
        model.training_metadata['mastery_curriculum']=dict(parent_checkpoint=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            updates=args.updates*(round_index+1),round_seed=81000+round_index,complex_count=count,
            source_replay=str(args.replay.resolve()),source_replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
            all_old_starts=args.all_old_starts,
            parent_replay=str(cache.resolve()),correction_beta=args.correction_beta,
            teacher_cruise=args.teacher_cruise,teacher_margin=args.teacher_margin,
            initial_complex=args.initial_complex,anchor_weight=args.anchor_weight,opening_weight=args.opening_weight,
            teacher_cases=[e['case'] for e in replay],anchors='same teacher targets on exact old routes; no competing source outputs',
            full_history=True,learning_rate=optimizer.param_groups[0]['lr'],development_seed=20001)
        path=args.out/f'candidate-{round_index}.pt';save_checkpoint(path,model,optimizer,[(e['x'],e['y']) for e in replay],payload['iteration'])
        fixed=validate(model)
        training=evaluate(model,[cases[e['case']] for e in complex_pool[:count]])
        development=evaluate(model,mixed_cases(20001,'selection',1))
        mastered=(fixed['successes']==7 and fixed['contacts']==0 and training['successes']==count and training['contacts']==0)
        report=dict(fixed=fixed,training=training,new=development,teacher_errors=errors,loss=loss,mastered=mastered,complex_count=count)
        write(args.out/f'candidate-{round_index}-development.json',report)
        summary=dict(round=round_index,complex_count=count,old=fixed['successes'],old_contacts=fixed['contacts'],training=training['successes'],training_contacts=training['contacts'],
                     new=development['successes'],new_contacts=development['contacts'],mastered=mastered,mean_teacher_mse=sum(e['mse'] for e in errors)/len(errors))
        results.append(summary);print(json.dumps(dict(phase='mastery',**summary)),flush=True)
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True))
        if mastered and count<len(complex_pool):count=min(count+2,len(complex_pool))
        elif sum(e['mse'] for e in errors)/len(errors)<.001 and optimizer.param_groups[0]['lr']>1e-4:
            for group in optimizer.param_groups:group['lr']=1e-4


if __name__=='__main__':main()
