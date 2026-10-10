"""Opt-in full-history learning through unchanged connectome topology.

Teacher labels and source distillation are training-only. All development
inference is the neural policy on exclusively observable inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import RecurrentPolicy,load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.validation import validate
from scripts.train_sequential import episode,evaluate
from scripts.improve_navigation import write,rank


def resolve_fit_scope(model,requested=None):
    saved=model.training_metadata.get('fit_scope',
        model.training_metadata.get('fixed_replay_refinement',{}).get('scope','all'))
    scope=requested if requested is not None else saved
    if scope not in ('all','sectors-residual'):
        raise ValueError('unknown fitting scope')
    return scope


def sequence_errors(predicted,labels,lengths,*,opening_weight=1.,supervision_starts=None):
    if opening_weight<1:
        raise ValueError('opening weight must be at least one')
    time=torch.arange(predicted.shape[1],device=predicted.device)[None,:]
    valid=time<lengths[:,None]
    if supervision_starts is not None:
        if supervision_starts.shape!=lengths.shape or torch.any(supervision_starts<0) or torch.any(supervision_starts>=lengths):
            raise ValueError('valid nonempty supervision windows required')
        valid=valid & (time>=supervision_starts[:,None])
    temporal=torch.where(time<10,opening_weight,1.)*valid
    error=((predicted-labels).square()*predicted.new_tensor([1.,2.])).mean(-1)
    stop=torch.where(labels[:,:,0].abs()<.02,3.,1.)
    return (error*stop*temporal).sum(1)/temporal.sum(1)


def fit_live_sequences(model,optimizer,episodes,anchors,*,updates=400,anchor_weight=4.,opening_weight=1.,scope='all'):
    if not model.trainable_core or not episodes or not anchors:
        raise ValueError('live core, labelled episodes and source anchors required')
    if scope not in ('all','sectors-residual'):
        raise ValueError('unknown fitting scope')
    if scope=='sectors-residual' and (model.encoder.in_features!=31 or model.motor_head!='residual32'):
        raise ValueError('sector schema and residual head required')
    original_encoder=model.encoder.weight[:,:19].detach().clone() if scope=='sectors-residual' else None
    pools={tier:[e for e in episodes if e['tier']==tier] for tier in ('simple','complex')}
    if any(not pool for pool in pools.values()):raise ValueError('both difficulty tiers required')
    losses=[]
    for step in range(updates):
        group=[];weights=[]
        for tier in ('simple','complex'):
            for _ in range(2):
                item=pools[tier][int(torch.randint(len(pools[tier]),()))]
                group.append((item['x'],item['y'],item.get('supervision_start',0)));weights.append(1.)
        for _ in range(2):
            ax,ay=anchors[int(torch.randint(len(anchors),()))];group.append((ax,ay,0));weights.append(anchor_weight)
        lengths=torch.tensor([len(x) for x,y,start in group])
        x=torch.nn.utils.rnn.pad_sequence([x for x,y,start in group],batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence([y for x,y,start in group],batch_first=True)
        optimizer.zero_grad()
        predicted,_=model(x)
        # Each episode contributes independently of its length; padding never
        # contributes. Entire histories are differentiated, without stale states.
        per_episode=sequence_errors(predicted,y,lengths,opening_weight=opening_weight,supervision_starts=torch.tensor([start for x,y,start in group]))
        loss=(per_episode*torch.tensor(weights)).sum()/sum(weights)
        loss.backward()
        if scope=='sectors-residual':
            for name,parameter in model.named_parameters():
                if name=='encoder.weight':parameter.grad[:,:19]=0
                elif not name.startswith('readout.residual.'):parameter.grad=None
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if original_encoder is not None:
            # Old optimizer momentum must not move protected columns either.
            with torch.no_grad():model.encoder.weight[:,:19].copy_(original_encoder)
        losses.append(float(loss.detach()))
        if (step+1)%50==0:
            print(json.dumps(dict(phase='fit_live',update=step+1,loss=losses[-1])),flush=True)
    return sum(losses[-min(50,len(losses)):])/min(50,len(losses))


def load_live_source(path, graph):
    model,_,_,_=load_checkpoint(path,graph,baseline=False)
    if model.motor_head!='residual32':
        raise ValueError('anatomical residual32 source required')
    expected_mask=RecurrentPolicy(graph).mask
    if not torch.equal(model.mask,expected_mask):
        raise ValueError('source mask differs from anatomical graph')
    model.trainable_core=True
    for parameter in model.parameters():parameter.requires_grad_(True)
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--rehearse-old',action='store_true')
    parser.add_argument('--anchor-weight',type=float,default=4.)
    parser.add_argument('--lr',type=float,default=1e-4)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    payload=torch.load(args.source,weights_only=True)
    canonical_path=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt')
    original=torch.load(canonical_path,weights_only=True)
    model=load_live_source(args.source,original['graph'])
    torch.set_num_threads(2);torch.manual_seed(100920261)
    # Keep the standard ordered optimizer group for checkpoint compatibility.
    # All existing neural parameters can learn; topology remains fixed.
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    mask=model.mask.clone()
    teacher_model,_,_,_=load_checkpoint(canonical_path,original['graph'])
    saved=torch.load('artifacts/imitation/experiments/sequential-transfer-20261008/live-anchors.pt',weights_only=True)
    if saved['source_sha256']!=hashlib.sha256(canonical_path.read_bytes()).hexdigest():raise ValueError('anchor mismatch')
    anchors=[]
    for x in saved['observations']:
        with torch.no_grad():y=teacher_model(x[None])[0][0]
        anchors.append((x,y.detach()))
    replay_source=args.replay or Path('artifacts/imitation/experiments/mixed-curriculum-20261009/trial/replay-2.pt')
    data=torch.load(replay_source,weights_only=True)
    replay=[dict(x=e['x'],y=e['y'],tier=e['tier'],case=e['case'],beta=e['beta'])
            for e in data['episodes'] if args.replay or e['beta']==1.]
    development=mixed_cases(20001,'selection',1)
    baseline=dict(fixed=validate(model),new=evaluate(model,development));write(args.out/'baseline-development.json',baseline)
    collection=[];candidates=[]
    if args.rehearse_old:
        from fly_brain_rc.imitation.curriculum import validation_cases
        for case in validation_cases():
            for beta in (1.,0.):
                x,y,r=episode(model,case,(0.,0.,0.),beta=beta,steps=2400)
                replay.append(dict(x=x,y=y,tier='simple',case=case.name,beta=beta))
                r.update(stage=-1,tier='simple',old_rehearsal=True);collection.append(r)
    for stage in range(3):
        if stage:
            for case,pose,tier in mixed_cases(21000+stage,'train',stage):
                for beta in (1.,0.):
                    try:
                        x,y,r=episode(model,case,pose,beta=beta,steps=2400 if beta==1 else 1600)
                        replay.append(dict(x=x,y=y,tier=tier,case=case.name,beta=beta));r.update(stage=stage,tier=tier);collection.append(r)
                    except ValueError as error:
                        collection.append(dict(stage=stage,scenario=case.name,beta=beta,error=str(error)))
        cache=args.out/f'replay-{stage}.pt';torch.save(dict(episodes=replay,anchors=anchors),cache)
        loss=fit_live_sequences(model,optimizer,replay,anchors,updates=400,anchor_weight=args.anchor_weight)
        assert torch.equal(model.mask,mask)
        assert torch.all(model.effective_weights()[mask==0]==0)
        model.training_metadata['live_core']=dict(source_sha256=source_hash,source_iteration=payload['iteration'],
            updates=400*(stage+1),development_seed=20001,new_training_seeds=list(range(21001,21001+stage)),
            replay=str(cache.resolve()),source_replay=str(replay_source.resolve()),parent_checkpoint=str(args.source.resolve()),
            optimizer_reset=True,old_rehearsal=args.rehearse_old,
            anchor_weight=args.anchor_weight,learning_rate=args.lr,full_history=True,collection=collection)
        path=args.out/f'candidate-{stage}.pt';save_checkpoint(path,model,optimizer,payload['trajectories'],payload['iteration'])
        fixed=validate(model);new=evaluate(model,development)
        r=dict(fixed=fixed,new=new,loss=loss);write(args.out/f'candidate-{stage}-development.json',r);candidates.append((path,r))
        print(json.dumps(dict(phase='candidate',stage=stage,old=fixed['successes'],old_contacts=fixed['contacts'],
                             new=new['successes'],new_contacts=new['contacts'])),flush=True)
    result=dict(candidates=[dict(path=str(p.resolve()),old=r['fixed']['successes'],old_contacts=r['fixed']['contacts'],
                                new=r['new']['successes'],new_contacts=r['new']['contacts']) for p,r in candidates],
                source_unchanged=hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash,
                topology_unchanged=True,deployed=False,final_not_used=True)
    write(args.out/'result.json',result)


if __name__=='__main__':main()
