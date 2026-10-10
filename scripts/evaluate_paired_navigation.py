"""Predeclared read-only comparison on fresh paired navigation maps."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.audit_neural_recovery import load
from scripts.improve_navigation import write
from scripts.prepare_neural_rehearsal import case_spec
from scripts.train_sequential import episode,evaluate


def clean(row):return bool(row['success'] and row['contacts']==0)


def paired_summary(source,candidate):
    def identity(r):return r['scenario'],r['family'],tuple(r.get('start_pose',()))
    if len(source)!=len(candidate) or any(identity(a)!=identity(b) for a,b in zip(source,candidate)):
        raise ValueError('paired case identities differ')
    if len({identity(r) for r in source})!=len(source):raise ValueError('duplicate paired cases')
    result={}
    for tier in ('simple','complex'):
        pairs=[(a,b) for a,b in zip(source,candidate) if a['family']==tier]
        gained=sum(not clean(a) and clean(b) for a,b in pairs);lost=sum(clean(a) and not clean(b) for a,b in pairs)
        n=gained+lost
        result[tier]=dict(gained=gained,lost=lost,new_contact_maps=sum(a['contacts']==0 and b['contacts']>0 for a,b in pairs),
            paired_exact_p=min(1.,2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n) if n else 1.)
        for name,index in [('source',0),('candidate',1)]:
            rows=[p[index] for p in pairs]
            result[tier][name]=dict(episodes=len(rows),arrivals=sum(r['success'] for r in rows),clean=sum(clean(r) for r in rows),
                contacts=sum(r['contacts'] for r in rows),contact_maps=sum(r['contacts']>0 for r in rows),
                stalled_seconds=sum(r['stalled_seconds'] for r in rows))
    return result


def practical_gate(old,previous,summary):
    old_rows=old['reports'];prior=[r for r in previous['reports'] if r['family']=='complex']
    return (len(old_rows)==7 and all(clean(r) for r in old_rows) and len(prior)==6 and all(clean(r) for r in prior)
        and summary['complex']['candidate']['clean']>summary['complex']['source']['clean']
        and summary['simple']['candidate']['clean']>=summary['simple']['source']['clean']
        and all(summary[t]['candidate']['contacts']<=summary[t]['source']['contacts'] for t in ('simple','complex')))


def observed_tail(model,x):
    with torch.no_grad():actions=model(x[None])[0][0,-20:]
    tail=x[-20:]
    return dict(target_visible_fraction=float(tail[:,0].mean()),last_target_visible=bool(tail[-1,0]>.5),
        mean_abs_throttle=float(actions[:,0].abs().mean()),mean_abs_steering=float(actions[:,1].abs().mean()),
        last_ranges_m=(tail[-1,6:12]*1.2).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    paths=dict(source=Path('artifacts/imitation/versions/main-23of32.pt'),candidate=Path('artifacts/imitation/experiments/favourite-recovery-distillation-20261009/live-3.pt'))
    expected=dict(source='e7155c9a2104a61f2936989a7d0b5b8bd2902e2e09ed2f1925c7c018958b5702',candidate='b419701511c6a0296f4dc49bd7df223f552c817233596e75ab4feef7fe7f6356')
    protected=list(paths.values())+[Path('artifacts/imitation/connectome.pt'),Path('artifacts/imitation/connectome.best.pt')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    if any(hashes[str(path)]!=expected[name] for name,path in paths.items()):raise ValueError('preselected checkpoint changed')
    seeds=list(range(1050001,1050017));cases=[c for s in seeds for c in mixed_cases(s,'final',2)]
    layouts=[case_spec(c,p,t) for c,p,t in cases];write(args.out/'layouts.json',layouts)
    protocol=dict(models={n:dict(path=str(p.resolve()),sha256=expected[n]) for n,p in paths.items()},protected_sha256=hashes,
        final_seeds=seeds,split='final',stage=2,steps=2400,episodes=192,simple=64,complex=128,
        layouts_sha256=hashlib.sha256((args.out/'layouts.json').read_bytes()).hexdigest(),
        comparison='new user-authorised practical independent comparison; does not replace the earlier strict training gate',
        practical_gate='old7/priorcomplex6 clean; complexclean strictly improves; simpleclean retained; contacts do not increase within either tier',
        statistics='two-sided exact paired clean-outcome test, descriptive only; no significance promotion gate',
        runtime='ordinary autonomous neural policy; no teacher or learning',earlier_reserved_1040001_1040008_used=False,
        scope='new seeds within existing generator; not arbitrary mazes, search tasks or hardware',automatic_promotion=False)
    write(args.out/'protocol.json',protocol);(args.out/'run.py').write_text(Path(__file__).read_text())
    reports={};legacy={}
    for name,path in paths.items():
        model=load(path);before={n:v.clone() for n,v in model.state_dict().items()}
        legacy[name]=dict(old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]),previous=evaluate(model,mixed_cases(20001,'selection',1)))
        reports[name]=[]
        for case,pose,tier in cases:
            x,_,report=episode(model,case,pose);report.update(family=tier,tail=observed_tail(model,x))
            reports[name].append(report);write(args.out/f'{name}.json',dict(reports=reports[name],legacy=legacy[name]))
        assert all(torch.equal(model.state_dict()[n],v) for n,v in before.items())
    summary=paired_summary(reports['source'],reports['candidate']);pairs=[]
    for a,b in zip(reports['source'],reports['candidate']):
        pairs.append(dict(case=a['scenario'],tier=a['family'],source_clean=clean(a),candidate_clean=clean(b),source=a,candidate=b))
    failure_diagnostics={}
    for name,rows in reports.items():
        failures=[r for r in rows if not clean(r)]
        failure_diagnostics[name]=dict(not_clean=len(failures),raw_arrival_with_contact=sum(r['success'] and r['contacts']>0 for r in failures),
            entered_goal_without_stable_arrival=sum(not r['success'] and r['min_goal_distance']<=.5 for r in failures),
            target_absent_entire_last20=sum(r['tail']['target_visible_fraction']==0 for r in failures),
            with_contacts=sum(r['contacts']>0 for r in failures),stalled_at_least_3s=sum(r['stalled_seconds']>=3 for r in failures))
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in protected);assert unchanged
    result=dict(protocol=protocol,summary=summary,legacy=legacy,pairs=pairs,failure_diagnostics=failure_diagnostics,
        practical_comparison_passed=practical_gate(legacy['candidate']['old'],legacy['candidate']['previous'],summary),
        protected_unchanged=unchanged,weights_unchanged=True,deployed=False,completed=True)
    write(args.out/'result.json',result);print(json.dumps(dict(phase='paired_final',summary=summary,practical_comparison_passed=result['practical_comparison_passed'])),flush=True)


if __name__=='__main__':main()
