"""Controlled fixed-replay residual fitting with successful old-output anchors."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.train_neural_ppo import freeze_actor
from scripts.train_mixed_curriculum import fit_mixture
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','anchors','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--anchor-weight',type=float,default=32.)
    parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--updates',type=int,default=2000)
    args=parser.parse_args()
    if not 0<args.anchor_weight<float('inf') or args.rounds<1 or args.updates<1:raise ValueError('invalid fitting settings')
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'run.py').write_text(Path(__file__).read_text())
    (args.out/'fit.snapshot.py').write_text(Path('scripts/train_mixed_curriculum.py').read_text())
    torch.set_num_threads(2);source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    saved=torch.load(args.source,weights_only=True);model,*rest=load_checkpoint(args.source,saved['graph']);freeze_actor(model)
    cache=torch.load(args.replay,weights_only=True)
    if cache['source_sha256']!=source_hash:raise ValueError('cached states belong to a different brain')
    replay=cache['episodes'];ah=torch.load(args.anchors,weights_only=True);anchors=(ah['h'],ah['y'])
    if sum(r['success'] for r in ah['reports'])!=7 or sum(r['contacts'] for r in ah['reports']):raise ValueError('anchors must be successful old seven')
    with torch.no_grad():
        if not torch.allclose(model.readout(anchors[0]).tanh(),anchors[1],atol=1e-6,rtol=0):raise ValueError('anchor output mismatch')
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);results=[]
    for stage in range(args.rounds):
        torch.manual_seed(130000+stage)
        losses=fit_mixture(model,optimizer,replay,anchors,complex_fraction=.5,updates=args.updates,anchor_weight=args.anchor_weight)
        assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in frozen.items())
        model.training_metadata.pop('fit_scope',None)
        model.training_metadata['cached_motor_refinement']=dict(source=str(args.source.resolve()),source_sha256=source_hash,
            replay=str(args.replay.resolve()),replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
            anchors=str(args.anchors.resolve()),anchors_sha256=hashlib.sha256(args.anchors.read_bytes()).hexdigest(),
            fit_scope='readout.residual only',core_frozen_verified=True,anchor_weight=args.anchor_weight,learning_rate=3e-4,
            updates=args.updates*(stage+1),seeds=list(range(130000,130001+stage)),fixed_replay=True,complex_fraction=.5)
        save_checkpoint(args.out/f'candidate-{stage}.pt',model,optimizer,[],saved['iteration'])
        old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);new=evaluate(model,mixed_cases(20001,'selection',1))
        complex_reports=[r for r in new['reports'] if r['family']=='complex']
        summary=dict(stage=stage,old=old['successes'],old_contacts=old['contacts'],new=new['successes'],new_contacts=new['contacts'],
            complex=sum(r['success'] for r in complex_reports),complex_contacts=sum(r['contacts'] for r in complex_reports))
        results.append(summary);write(args.out/f'candidate-{stage}-development.json',dict(old=old,new=new,losses=losses,summary=summary))
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True))
        print(json.dumps(dict(phase='cached_motor_candidate',**summary)),flush=True)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash


if __name__=='__main__':main()
