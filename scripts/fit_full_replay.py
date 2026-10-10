"""Full-batch route fitting diagnostic with an external LBFGS optimizer.

Runtime is unchanged. The ordinary brain checkpoint has an explicitly fresh
Adam optimizer for loader compatibility; actual LBFGS state is saved separately.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_live_core import load_live_source,sequence_errors
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write
from fly_brain_rc.imitation.model import save_checkpoint
from fly_brain_rc.imitation.validation import validate
from fly_brain_rc.imitation.navigation_cases import mixed_cases


class FullReplayFitter:
    def __init__(self,model,episodes,*,opening_weight=10.):
        if not episodes or not model.trainable_core:raise ValueError('trainable model and replay required')
        counts={tier:sum(e['tier']==tier for e in episodes) for tier in ('simple','complex')}
        if any(not n for n in counts.values()):raise ValueError('both tiers required')
        if any(len(e['x'])!=len(e['y']) or not len(e['x']) for e in episodes):raise ValueError('aligned nonempty histories required')
        self.model=model;self.opening_weight=opening_weight
        self.x=torch.nn.utils.rnn.pad_sequence([e['x'] for e in episodes],batch_first=True)
        self.y=torch.nn.utils.rnn.pad_sequence([e['y'] for e in episodes],batch_first=True).clone()
        self.y[:,:,1].clamp_(-.98,.98)
        self.lengths=torch.tensor([len(e['x']) for e in episodes])
        self.weights=torch.tensor([.5/counts[e['tier']] for e in episodes])
        self.optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=5,history_size=20,line_search_fn='strong_wolfe')
        self.evaluations=0
    def loss(self):
        predicted,_=self.model(self.x)
        return (sequence_errors(predicted,self.y,self.lengths,opening_weight=self.opening_weight)*self.weights).sum()
    def step(self):
        def closure():
            self.optimizer.zero_grad();loss=self.loss()
            if not torch.isfinite(loss):raise ValueError('nonfinite optimizer loss')
            loss.backward();self.evaluations+=1
            return loss
        self.optimizer.step(closure)
        with torch.no_grad():return float(self.loss())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args()
    if args.rounds<1 or args.steps<1:raise ValueError('positive optimizer counts required')
    args.out.mkdir(exist_ok=False,parents=True);(args.out/'run.py').write_text(Path(__file__).read_text())
    torch.set_num_threads(2)
    graph=torch.load('artifacts/imitation/experiments/navigation-20261008/baseline.pt',weights_only=True)['graph']
    model=load_live_source(args.source,graph);payload=torch.load(args.source,weights_only=True)
    episodes=torch.load(args.replay,weights_only=True)['episodes']
    if any(e['x'].shape[1]!=model.encoder.in_features for e in episodes):raise ValueError('replay schema mismatch')
    fitter=FullReplayFitter(model,episodes);mask=model.mask.clone();results=[]
    known={c.name:(c,p,t) for seed in range(31000,31004) for c,p,t in mixed_cases(seed,'train',1)}
    names=list(dict.fromkeys(e['case'] for e in episodes if e['tier']=='complex'))
    for stage in range(args.rounds):
        for step in range(args.steps):
            loss=fitter.step()
            if (step+1)%5==0:print(json.dumps(dict(phase='full_replay_fit',stage=stage,step=step+1,loss=loss,evaluations=fitter.evaluations)),flush=True)
        assert torch.equal(mask,model.mask)
        external=args.out/f'optimizer-{stage}.pt';torch.save(fitter.optimizer.state_dict(),external)
        model.training_metadata['fit_scope']='all'
        model.training_metadata['full_replay_fit']=dict(parent_checkpoint=str(args.source.resolve()),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            replay=str(args.replay.resolve()),replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
            optimizer='LBFGS',optimizer_state=str(external.resolve()),brain_checkpoint_optimizer='fresh Adam; actual LBFGS stored separately',
            outer_steps=args.steps*(stage+1),closure_evaluations=fitter.evaluations,steering_target_clamp=.98,opening_weight=10.,full_history=True)
        path=args.out/f'candidate-{stage}.pt'
        save_checkpoint(path,model,torch.optim.Adam(model.parameters(),lr=3e-4),[(e['x'],e['y']) for e in episodes],payload['iteration'])
        fixed=validate(model);training=evaluate(model,[known[n] for n in names]);new=evaluate(model,mixed_cases(20001,'selection',1))
        write(args.out/f'candidate-{stage}-development.json',dict(fixed=fixed,training=training,new=new,loss=loss))
        summary=dict(stage=stage,loss=loss,old=fixed['successes'],old_contacts=fixed['contacts'],training=training['successes'],training_contacts=training['contacts'],new=new['successes'],new_contacts=new['contacts'])
        results.append(summary);print(json.dumps(dict(phase='full_replay_candidate',**summary)),flush=True)
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True))


if __name__=='__main__':main()
