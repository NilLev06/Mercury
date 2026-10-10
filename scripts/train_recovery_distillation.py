"""Bounded recovery follow-up with favourite rehearsal labels and balanced sampling."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.model import save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.audit_neural_recovery import load
from scripts.improve_navigation import write
from scripts.prepare_neural_rehearsal import restore_case
from scripts.train_live_core import load_live_source,fit_live_sequences
from scripts.train_neural_ppo import freeze_actor
from scripts.train_precontact_dagger import expanded_gate
from scripts.train_sequential import evaluate
from scripts.train_validated_recovery import counts


def distilled_replay(model,fixed,recoveries):
    summary={}
    for tier in ('simple','complex'):
        base=sum(e['tier']==tier for e in fixed);new=sum(e['tier']==tier for e in recoveries)
        if not base or not new:raise ValueError('rehearsal and recoveries required for both tiers')
        repeats=math.ceil(base/new)
        summary[tier]=dict(base=base,recoveries=new,repeats=repeats,recovery_fraction=new*repeats/(base+new*repeats))
    replay=[]
    for e in fixed:
        with torch.no_grad():labels=model(e['x'][None])[0][0].detach().clone()
        replay.append(dict({k:v for k,v in e.items() if k not in ('h','y')},y=labels,teaching_actor='favourite_distillation'))
    for e in recoveries:replay.extend([e]*summary[e['tier']]['repeats'])
    return replay,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--stages',type=int,default=4);parser.add_argument('--updates',type=int,default=200)
    args=parser.parse_args()
    if min(args.stages,args.updates)<1:raise ValueError('positive counts required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source_path=Path('artifacts/imitation/versions/main-23of32.pt')
    replay_path=Path('artifacts/imitation/experiments/broad-neural-replay-20261009/rehearsal.pt')
    recovery_folder=Path('artifacts/imitation/experiments/favourite-validated-recovery-20261009')
    protected=[source_path,Path('artifacts/imitation/connectome.pt'),Path('artifacts/imitation/connectome.best.pt')]
    protected_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    saved=torch.load(replay_path,weights_only=True);recoveries=torch.load(recovery_folder/'recoveries.pt',weights_only=True)
    source_sha=protected_sha[str(source_path)]
    if not saved['collection_complete'] or saved['source_sha256']!=source_sha or recoveries['source_sha256']!=source_sha:
        raise ValueError('source and training data differ')
    raw=torch.load(source_path,weights_only=True);source=load_live_source(source_path,raw['graph'])
    fixed=saved['episodes'];anchors=[(e['x'],e['y']) for e in fixed if e['teaching_actor']=='old_source']
    if len(anchors)!=21:raise ValueError('expected21 favourite anchors')
    replay,balance=distilled_replay(source,fixed,recoveries['episodes'])
    protocol=dict(source_sha256=source_sha,protected_sha256=protected_sha,stages=args.stages,updates_per_stage=args.updates,
        recovery_sha256=hashlib.sha256((recovery_folder/'recoveries.pt').read_bytes()).hexdigest(),
        legacy_replay_sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),balance=balance,anchor_weight=4.,
        learning_rates=dict(frozen=1e-4,live=1e-5),fit_seeds=[1100001+i for i in range(args.stages)],
        development_seeds=list(range(1030001,1030009)),reserved_final_seeds=list(range(1040001,1040009)),
        gate='old7/priorcomplex6 clean; complex>=58/64 and zero contacts all64; simple>=30/32 clean contacts<=515',
        rehearsal_labels='full-history favourite outputs on legacy observable histories',teacher='training recoveries only',
        runtime='unchanged autonomous neural evaluator',full_history=True,cached_states_used=False)
    write(args.out/'protocol.json',protocol);(args.out/'run.py').write_text(Path(__file__).read_text())
    (args.out/'fit.snapshot.py').write_text(Path('scripts/train_live_core.py').read_text())
    torch.save(dict(episodes=replay,source_sha256=source_sha,balance=balance),args.out/'replay.pt')
    development=[v for seed in range(1030001,1030009) for v in mixed_cases(seed,'selection',2)]
    training=[restore_case(saved['case_specs'][e['case']]) for e in recoveries['episodes']]
    baseline=json.loads((recovery_folder/'baseline-development.json').read_text())
    source_report=json.loads((recovery_folder/'source-development.json').read_text())
    # Verify report provenance against the protocol written before collection.
    earlier=json.loads((recovery_folder/'protocol.json').read_text())
    if earlier['source_sha256']!=source_sha or earlier['development_seeds']!=protocol['development_seeds']:
        raise ValueError('source report provenance differs')
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
            model.training_metadata['recovery_distillation']=dict(protocol,arm=arm,stage=stage,core_frozen_verified=arm=='frozen')
            model.trainable_core=arm=='live'
            path=args.out/f'{arm}-{stage}.pt';save_checkpoint(path,model,optimizer,[],raw['iteration']);model.trainable_core=True
            old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()])
            previous=evaluate(model,mixed_cases(20001,'selection',1));expanded=evaluate(model,development)
            probe=evaluate(model,training)
            result=dict(arm=arm,stage=stage,updates=(stage+1)*args.updates,loss=loss,
                old_clean=sum(r['success'] and r['contacts']==0 for r in old['reports']),old_contacts=old['contacts'],
                prior_complex_clean=sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in previous['reports']),
                expanded=counts(expanded),training_probe=counts(probe),development_gate=expanded_gate(old,previous,expanded,baseline),
                path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            results.append(result)
            write(args.out/f'{arm}-{stage}.json',dict(summary=result,old=old,previous=previous,expanded=expanded,training_probe=probe))
            unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==protected_sha[str(p)] for p in protected)
            assert unchanged
            write(args.out/'result.json',dict(source=counts(source_report),candidates=results,protected_unchanged=unchanged,
                deployed=False,final_not_used=True))
            print(json.dumps(dict(phase='distillation_candidate',**result)),flush=True)


if __name__=='__main__':main()
