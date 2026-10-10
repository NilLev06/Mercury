"""Mixed teacher/student curriculum; preserve source and deploy nothing.

Every stage includes simple and complex routes. Final maps are not read until
one candidate and its residual scale have been selected on development maps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.validation import validate
from scripts.train_sequential import episode,evaluate
from scripts.improve_navigation import neural_states,build_optimizer,blend_residual,write,rank


def fit_mixture(model,optimizer,replay,anchors,*,complex_fraction,updates=2000,anchor_weight=1.):
    pools={tier:[item for item in replay if item['tier']==tier] for tier in ('simple','complex')}
    if any(not p for p in pools.values()) or not 0<complex_fraction<1 or not 0<anchor_weight<float('inf'):
        raise ValueError('both difficulty tiers and bounded mixture required')
    anchor_h,anchor_y=anchors
    for _ in range(updates):
        h,y,weight=[],[],[]
        for tier,fraction in (('simple',1-complex_fraction),('complex',complex_fraction)):
            for _ in range(4):
                item=pools[tier][int(torch.randint(len(pools[tier]),()))]
                i=torch.randint(len(item['h']),(64,))
                h.append(item['h'][i]);y.append(item['y'][i])
                weight.append(torch.full((64,),fraction/4))
        h,y,weight=torch.cat(h),torch.cat(y),torch.cat(weight)
        prediction=torch.tanh(model.readout(h))
        stop=torch.where(y[:,0].abs()<.02,5.,1.)
        error=((prediction-y).square()*torch.tensor([1.,2.])).mean(-1)
        supervised=(error*stop*weight).sum()/weight.sum()
        a=torch.randint(len(anchor_h),(256,))
        distill=(torch.tanh(model.readout(anchor_h[a]))-anchor_y[a]).square().mean()
        loss=supervised+anchor_weight*distill
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.readout.residual.parameters(),1.)
        optimizer.step()
    return dict(supervised=float(supervised.detach()),distillation=float(distill.detach()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    payload=torch.load(args.source,weights_only=True)
    model,_,old,age=load_checkpoint(args.source,payload['graph'])
    if model.motor_head!='residual32':raise ValueError('residual32 required')
    torch.set_num_threads(2);torch.manual_seed(10092026)
    base={k:v.clone() for k,v in model.state_dict().items()}
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=build_optimizer(model,3e-4)
    # Source-authenticated actual successful old-policy episodes, captured by
    # the preceding experiment. Reconstruct states through the unchanged core.
    saved=torch.load('artifacts/imitation/experiments/sequential-transfer-20261008/live-anchors.pt',weights_only=True)
    if saved['source_sha256']!=source_hash:raise ValueError('anchor/source mismatch')
    ah,ay=[],[]
    for x in saved['observations']:
        h,y=neural_states(model,x);ah.append(h);ay.append(y)
    anchors=(torch.cat(ah),torch.cat(ay))
    selection=mixed_cases(14001,'selection',1)
    baseline_fixed=validate(model);baseline_new=evaluate(model,selection)
    write(args.out/'baseline-selection.json',dict(fixed=baseline_fixed,new=baseline_new))
    replay,collection,candidates=[],[],[]
    for stage in range(3):
        cases=mixed_cases(15001+stage,'train',stage)
        for case,pose,tier in cases:
            for beta in (1.,0.):
                try:
                    x,y,r=episode(model,case,pose,beta=beta,steps=2400 if beta==1 else 1600)
                    h,_=neural_states(model,x)
                    replay.append(dict(x=x,y=y,h=h,tier=tier,case=case.name,beta=beta))
                    r.update(stage=stage,tier=tier);collection.append(r)
                except ValueError as error:
                    r=dict(stage=stage,tier=tier,scenario=case.name,beta=beta,error=str(error))
                    collection.append(r);print(json.dumps(r),flush=True)
        write(args.out/'collection.json',collection)
        cache=args.out/f'replay-{stage}.pt'
        torch.save(dict(episodes=replay,anchor_states=anchors[0],anchor_outputs=anchors[1],source_sha256=source_hash),cache)
        losses=fit_mixture(model,optimizer,replay,anchors,complex_fraction=(1/3,1/2,2/3)[stage])
        assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
        model.training_metadata['mixed_curriculum']=dict(
            source_sha256=source_hash,source_iteration=age,train_seeds=list(range(15001,15002+stage)),
            selection_seed=14001,final_seeds=[16001,16002],retained_updates=2000*(stage+1),
            stage=stage,complex_fraction=(1/3,1/2,2/3)[stage],learning_rate=3e-4,
            anchor_weight=1,replay=str(cache.resolve()),collection=collection)
        path=args.out/f'candidate-{stage}.pt';save_checkpoint(path,model,optimizer,old,age)
        fixed=validate(model);new=evaluate(model,selection)
        trained=evaluate(model,[v for v in cases if v[2]=='complex'][:2])
        r=dict(fixed=fixed,new=new,training_probe=trained,losses=losses)
        write(args.out/f'candidate-{stage}-selection.json',r);candidates.append((path,r))
        print(json.dumps(dict(phase='candidate',stage=stage,old=fixed['successes'],contacts=fixed['contacts'],
                             new=new['successes'],new_contacts=new['contacts'],trained=trained['successes'])),flush=True)
    # Scale selection is part of development only; weights, not a runtime gate.
    options=[]
    for path,r in candidates:
        state=torch.load(path,weights_only=True)
        for scale in (.25,.5,.75,1.):
            if scale==1.:
                assessment=r
            else:
                model.load_state_dict(blend_residual(base,state['model'],scale))
                fixed=validate(model)
                assessment=dict(fixed=fixed)
                if fixed['successes']==7 and fixed['contacts']==0 and not fixed['cancelled']:
                    assessment['new']=evaluate(model,selection)
            write(args.out/f'{path.stem}-scale-{scale}.json',assessment)
            fixed=assessment['fixed']
            if (fixed['successes']==7 and fixed['contacts']==0 and not fixed['cancelled']
                    and 'new' in assessment and rank(assessment['new'])>rank(baseline_new)):
                options.append((path,scale,assessment))
    if options:
        path,scale,assessment=max(options,key=lambda v:rank(v[2]['new']))
        chosen=torch.load(path,weights_only=True)
        model.load_state_dict(blend_residual(base,chosen['model'],scale))
        model.training_metadata=chosen['training_metadata']
        model.training_metadata['parameter_blend']=dict(scale=scale,parent=str(path.resolve()),optimizer_reset=True)
        selected=args.out/'selected.pt';save_checkpoint(selected,model,build_optimizer(model,3e-4),old,age)
        final=mixed_cases(16001,'final',2)+mixed_cases(16002,'final',2)
        new=evaluate(model,final);model.load_state_dict(base);previous=evaluate(model,final)
        result=dict(selected=str(selected.resolve()),scale=scale,candidate_final=new,baseline_final=previous,
                    accepted=new['successes']>previous['successes'] and new['contacts']<=previous['contacts'])
    else:result=dict(selected=None,accepted=False,reason='No candidate retains original gate and improves development score')
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash
    result.update(source_sha256=source_hash,deployed=False,frozen_core_verified=True)
    write(args.out/'result.json',result)
    print(json.dumps(dict(phase='complete',selected=result['selected'],accepted=result['accepted'])),flush=True)


if __name__=='__main__':main()
