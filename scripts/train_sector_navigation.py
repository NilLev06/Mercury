"""Isolated richer camera input experiment, with assisted training only.

No live writes. Runtime decisions are exclusively the masked neural policy.
Original nineteen channels retain their meanings; all blue image components
contribute to twelve added perceptual channels. No privileged inference inputs.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint,upgrade_sector_inputs
from fly_brain_rc.imitation.observations import SCHEMA
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.validation import validate
from scripts.train_live_core import load_live_source,fit_live_sequences,resolve_fit_scope
from scripts.train_sequential import episode,evaluate
from scripts.improve_navigation import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=6)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--replay',type=Path)
    parser.add_argument('--anchor-weight',type=float,default=4.)
    parser.add_argument('--opening-weight',type=float,default=1.)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--assistance',type=float)
    parser.add_argument('--seed-base',type=int,default=32000)
    parser.add_argument('--scope',choices=('all','sectors-residual'))
    args=parser.parse_args()
    if (args.rounds<1 or args.updates<1 or args.anchor_weight<=0 or args.lr<=0
            or args.opening_weight<1 or (args.assistance is not None and not 0<=args.assistance<=1)):
        raise ValueError('invalid training parameters')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(100920263)
    canonical=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt')
    graph=torch.load(canonical,weights_only=True)['graph']
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    source=load_live_source(args.source,graph)
    model=upgrade_sector_inputs(source) if source.observation_schema==SCHEMA else source
    scope=resolve_fit_scope(model,args.scope)
    reference,_,_,_=load_checkpoint(canonical,graph)
    reference=upgrade_sector_inputs(reference)
    mask=model.mask.clone();optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    replay=[];anchors=[];collection=[];results=[]
    def collect(case,pose,tier,beta):
        try:
            x,y,r=episode(model,case,pose,beta=beta)
            replay.append(dict(x=x,y=y,tier=tier,case=case.name,beta=beta));r.update(tier=tier);collection.append(r)
        except ValueError as error:
            collection.append(dict(scenario=case.name,beta=beta,error=str(error)))
    if args.replay:
        saved=torch.load(args.replay,weights_only=True)
        replay,anchors=saved['episodes'],saved['anchors']
        if (any(e['x'].shape[1]!=model.encoder.in_features for e in replay)
                or any(x.shape[1]!=model.encoder.in_features for x,y in anchors)):
            raise ValueError('replay schema mismatch')
    else:
        for case in validation_cases():
            x,_,r=episode(reference,case,(0.,0.,0.))
            if not r['success'] or r['contacts']:raise RuntimeError('anchor baseline failed')
            with torch.no_grad():y=reference(x[None])[0][0]
            anchors.append((x,y.detach()))
            for pose in ((0.,0.,0.),(0.,.025,.035),(0.,-.025,-.035)):
                collect(case,pose,'simple',1.)
        # More varied, successful initial coverage; no failed old student buffers.
        for seed in range(31000,31004):
            for case,pose,tier in mixed_cases(seed,'train',1):collect(case,pose,tier,1.)
    for stage in range(args.rounds):
        if stage>=2 or args.replay:
            beta=args.assistance if args.assistance is not None else (.8,.6,.4,.2)[max(0,min(stage-2,3))]
            for case,pose,tier in mixed_cases(args.seed_base+stage,'train',1):
                collect(case,pose,tier,1.);collect(case,pose,tier,beta)
            # Correct nearby states on old maps rather than only distilling
            # commands along one exact successful trajectory.
            for case in validation_cases():collect(case,(0.,0.,0.),'simple',beta)
        cache=args.out/f'replay-{stage}.pt'
        torch.save(dict(episodes=replay,anchors=anchors,collection=collection),cache)
        torch.manual_seed(61000+stage)
        loss=fit_live_sequences(model,optimizer,replay,anchors,updates=args.updates,
                               anchor_weight=args.anchor_weight,opening_weight=args.opening_weight,scope=scope)
        assert torch.equal(model.mask,mask)
        assert torch.all(model.effective_weights()[mask==0]==0)
        model.training_metadata['fit_scope']=scope
        model.training_metadata['sector_navigation']=dict(source_sha256=source_hash,parent_checkpoint=str(args.source.resolve()),scope=scope,
            updates=args.updates*(stage+1),full_history=True,replay=str(cache.resolve()),anchor_weight=args.anchor_weight,learning_rate=args.lr,
            source_replay=str(args.replay.resolve()) if args.replay else None,optimizer_reset=True,
            opening_weight=args.opening_weight,assistance=args.assistance,training_seed_base=args.seed_base,
            warm_training_seeds=[] if args.replay else list(range(31000,31004)),
            assisted_training_seeds=list(range(args.seed_base+(0 if args.replay else 2),args.seed_base+stage+1)),
            development_seed=20001,old_maps_used_for_rehearsal=True)
        path=args.out/f'candidate-{stage}.pt'
        save_checkpoint(path,model,optimizer,[(e['x'],e['y']) for e in replay],args.updates*(stage+1))
        fixed=validate(model);new=evaluate(model,mixed_cases(20001,'selection',1))
        report=dict(fixed=fixed,new=new,loss=loss);write(args.out/f'candidate-{stage}-development.json',report)
        summary=dict(stage=stage,old=fixed['successes'],old_contacts=fixed['contacts'],new=new['successes'],new_contacts=new['contacts'])
        results.append(summary);print(json.dumps(dict(phase='candidate',**summary)),flush=True)
        write(args.out/'result.json',dict(candidates=results,source_unchanged=hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash,
                                         topology_unchanged=True,deployed=False,final_not_used=True))


if __name__=='__main__':main()
