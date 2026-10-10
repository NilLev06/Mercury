"""Offline capacity diagnostic, not a deployable brain or navigation score."""
import copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from fly_brain_rc.imitation.model import load_checkpoint
from scripts.improve_navigation import neural_states,write


def main():
    source=Path('artifacts/imitation/experiments/gentle-corrected-mastery-20261009/candidate-7.pt')
    replay=Path('artifacts/imitation/experiments/physical-mirror-replay-20261009/augmented.pt')
    out=Path('artifacts/imitation/experiments/motor-capacity-probe-matched-20261009');out.mkdir(parents=True,exist_ok=False)
    (out/'run.py').write_text(Path(__file__).read_text());torch.set_num_threads(2)
    raw=torch.load(source,weights_only=True);model,*_=load_checkpoint(source,raw['graph'])
    episodes=torch.load(replay,weights_only=True)['episodes'];data=[]
    for e in episodes:
        h,_=neural_states(model,e['x'])
        # Group reflected pairs together: two entire seeds remain out of fitting.
        heldout=e['tier']=='complex' and any('-'+str(s)+'-' in e['case'] for s in (31002,31003))
        data.append(dict(h=h,y=e['y'],tier=e['tier'],heldout=heldout,case=e['case']))
    results=[]
    for width in (32,128):
        torch.manual_seed(10261009)
        head=nn.Sequential(nn.Linear(len(model.graph['ids']),width),nn.Tanh(),nn.Linear(width,2))
        with torch.no_grad():
            head[0].weight[:32].copy_(model.readout.residual[0].weight);head[0].bias[:32].copy_(model.readout.residual[0].bias)
            head[2].weight.zero_();head[2].weight[:,:32].copy_(model.readout.residual[2].weight);head[2].bias.copy_(model.readout.residual[2].bias)
        base=copy.deepcopy(model.readout.base)
        for p in base.parameters():p.requires_grad_(False)
        pools={t:[e for e in data if e['tier']==t and not e['heldout']] for t in ('simple','complex')}
        opt=torch.optim.Adam(head.parameters(),lr=3e-4)
        # Identical episode/frame draws despite differing initialization sizes.
        torch.manual_seed(10261010)
        for step in range(3000):
            hs=[];ys=[]
            for t in ('simple','complex'):
                for _ in range(4):
                    e=pools[t][int(torch.randint(len(pools[t]),()))];idx=torch.randint(len(e['h']),(64,));hs.append(e['h'][idx]);ys.append(e['y'][idx])
            h,y=torch.cat(hs),torch.cat(ys);pred=(base(h)+head(h)).tanh()
            error=((pred-y).square()*torch.tensor([1.,2.])).mean(-1);loss=(error*torch.where(y[:,0].abs()<.02,3.,1.)).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            if (step+1)%500==0:
                row=dict(width=width,updates=step+1)
                for label,subset in [('train',[e for e in data if not e['heldout']]),('heldout_complex',[e for e in data if e['heldout']])]:
                    errors=[]
                    with torch.no_grad():
                        for e in subset:errors.append(float(((base(e['h'])+head(e['h'])).tanh()-e['y']).square().mean()))
                    row[label]=sum(errors)/len(errors)
                results.append(row);write(out/'result.json',dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),probe_only=True,autonomous_navigation_not_evaluated=True,heldout_seeds=[31002,31003],results=results))
                print(json.dumps(row),flush=True)
        torch.save(dict(head=head.state_dict(),base=base.state_dict(),width=width),out/f'offline-head-{width}.pt')


if __name__=='__main__':main()
