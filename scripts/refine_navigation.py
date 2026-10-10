"""Controlled full-history fitting of a fixed saved replay, no live writes."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import save_checkpoint
from fly_brain_rc.imitation.validation import validate
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.train_live_core import load_live_source,fit_live_sequences,resolve_fit_scope
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--opening-weight',type=float,default=1.)
    parser.add_argument('--anchor-weight',type=float,default=4.)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--scope',choices=('all','sectors-residual'))
    args=parser.parse_args()
    if args.opening_weight<1 or args.anchor_weight<=0 or args.lr<=0 or args.updates<1 or args.rounds<1:raise ValueError('invalid fitting parameters')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    graph=torch.load('artifacts/imitation/experiments/navigation-20261008/baseline.pt',weights_only=True)['graph']
    model=load_live_source(args.source,graph)
    scope=resolve_fit_scope(model,args.scope)
    replay=torch.load(args.replay,weights_only=True)
    episodes,anchors=replay['episodes'],replay['anchors']
    if any(e['x'].shape[1]!=model.encoder.in_features for e in episodes) or any(x.shape[1]!=model.encoder.in_features for x,y in anchors):raise ValueError('replay schema does not match source')
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest();replay_hash=hashlib.sha256(args.replay.read_bytes()).hexdigest()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    mask=model.mask.clone();results=[]
    for stage in range(args.rounds):
        torch.manual_seed(51000+stage)
        loss=fit_live_sequences(model,optimizer,episodes,anchors,updates=args.updates,anchor_weight=args.anchor_weight,opening_weight=args.opening_weight,scope=scope)
        assert torch.equal(mask,model.mask)
        model.training_metadata['fit_scope']=scope
        model.training_metadata['fixed_replay_refinement']=dict(parent_checkpoint=str(args.source.resolve()),source_sha256=source_hash,
            replay=str(args.replay.resolve()),replay_sha256=replay_hash,optimizer_reset=True,
            updates=args.updates*(stage+1),seed_by_stage=list(range(51000,51001+stage)),opening_weight=args.opening_weight,
            opening_steps=10,anchor_weight=args.anchor_weight,learning_rate=args.lr,scope=scope,full_history=True,development_seed=20001)
        path=args.out/f'candidate-{stage}.pt';save_checkpoint(path,model,optimizer,[(e['x'],e['y']) for e in episodes],args.updates*(stage+1))
        fixed=validate(model);new=evaluate(model,mixed_cases(20001,'selection',1));write(args.out/f'candidate-{stage}-development.json',dict(fixed=fixed,new=new,loss=loss))
        summary=dict(stage=stage,old=fixed['successes'],old_contacts=fixed['contacts'],new=new['successes'],new_contacts=new['contacts']);results.append(summary);print(json.dumps(dict(phase='candidate',**summary)),flush=True)
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True,source_unchanged=source_hash==hashlib.sha256(args.source.read_bytes()).hexdigest(),replay_unchanged=replay_hash==hashlib.sha256(args.replay.read_bytes()).hexdigest()))

if __name__=='__main__':main()
