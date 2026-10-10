"""Train a final neural throttle row while preserving steering exactly."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.train_sequential import evaluate
from scripts.train_precontact_dagger import expanded_gate
from scripts.improve_navigation import write


def fit_throttle(model,replay,anchors,*,updates=3000,anchor_weight=1.):
    if updates<1 or not 0<anchor_weight<float('inf'):raise ValueError('positive fitting settings required')
    for p in model.parameters():p.requires_grad_(False)
    layer=model.readout.residual[-1]
    layer.weight.requires_grad_(True);layer.bias.requires_grad_(True)
    optimizer=torch.optim.Adam([layer.weight,layer.bias],lr=3e-4)
    with torch.no_grad():
        h=torch.cat([e['h'] for e in replay]);y=torch.cat([e['y'][:,0] for e in replay])
        features=model.readout.residual[:2](h);base=model.readout.base(h)[:,0]
        ah,ay=anchors;af=model.readout.residual[:2](ah);ab=model.readout.base(ah)[:,0];ay=ay[:,0]
    original_weight=layer.weight[1].detach().clone();original_bias=layer.bias[1].detach().clone()
    for _ in range(updates):
        i=torch.randint(len(h),(512,));a=torch.randint(len(ah),(256,))
        prediction=torch.tanh(base[i]+features[i]@layer.weight[0]+layer.bias[0])
        anchor=torch.tanh(ab[a]+af[a]@layer.weight[0]+layer.bias[0])
        loss=((prediction-y[i]).square()*torch.where(y[i].abs()<.02,5.,1.)).mean()+anchor_weight*(anchor-ay[a]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert torch.equal(layer.weight[1],original_weight) and torch.equal(layer.bias[1],original_bias)
    return float(loss.detach())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('artifacts/imitation/experiments/navigation-20261008/baseline.pt'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);(args.out/'run.py').write_text(Path(__file__).read_text())
    torch.set_num_threads(2);raw=torch.load(args.source,weights_only=True);model,*rest=load_checkpoint(args.source,raw['graph'])
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest();base={k:v.clone() for k,v in model.state_dict().items()}
    saved=torch.load(args.replay,weights_only=True)
    if saved['source_sha256']!=source_hash:raise ValueError('source cache mismatch')
    replay=[e for e in saved['episodes'] if e['teaching_actor'] in ('baseline','old_source','source_complex')]
    anchors=(saved['anchors']['h'],saved['anchors']['y'])
    expanded=[v for seed in range(930001,930005) for v in mixed_cases(seed,'selection',2)]
    baseline_raw=torch.load(args.baseline,weights_only=True);baseline,*_=load_checkpoint(args.baseline,baseline_raw['graph'])
    baseline_report=evaluate(baseline,expanded);write(args.out/'baseline-expanded.json',baseline_report)
    summaries=[]
    for anchor_weight in (1.,4.,16.):
        model.load_state_dict(base);torch.manual_seed(960001)
        loss=fit_throttle(model,replay,anchors,anchor_weight=anchor_weight);trained={k:v.clone() for k,v in model.state_dict().items()}
        for scale in (.5,1.):
            model.load_state_dict({k:base[k]+scale*(v-base[k]) if k in ('readout.residual.2.weight','readout.residual.2.bias') else v for k,v in trained.items()})
            for k,v in model.state_dict().items():
                if k not in ('readout.residual.2.weight','readout.residual.2.bias'):assert torch.equal(v,base[k])
                else:assert torch.equal(v[1],base[k][1])
            model.training_metadata['throttle_calibration']=dict(source_sha256=source_hash,replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
                anchor_weight=anchor_weight,scale=scale,updates=3000,seed=960001,steering_row_verified_unchanged=True,fit_scope='final throttle row only')
            name=f'anchor-{anchor_weight:g}-scale-{scale:g}'
            # Standard loader restores the conventional frozen-core residual scope;
            # optimizer is explicitly reset rather than serializing a partial one.
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            save_checkpoint(args.out/(name+'.pt'),model,optimizer,[],raw['iteration'])
            old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);previous=None;assessment=None;gate=False
            if old['successes']==7 and old['contacts']==0:
                previous=evaluate(model,mixed_cases(20001,'selection',1))
                complex_old=[r for r in previous['reports'] if r['family']=='complex']
                if all(r['success'] and r['contacts']==0 for r in complex_old):
                    assessment=evaluate(model,expanded);gate=expanded_gate(old,previous,assessment,baseline_report)
            summary=dict(name=name,old=old['successes'],old_contacts=old['contacts'],previous_complex=None if previous is None else sum(r['success'] and r['contacts']==0 for r in previous['reports'] if r['family']=='complex'),
                expanded=None if assessment is None else {tier:dict(clean=sum(r['success'] and r['contacts']==0 for r in assessment['reports'] if r['family']==tier),contacts=sum(r['contacts'] for r in assessment['reports'] if r['family']==tier)) for tier in ('simple','complex')},development_gate=gate,loss=loss)
            summaries.append(summary);write(args.out/(name+'.json'),dict(summary=summary,old=old,previous=previous,expanded=assessment))
            write(args.out/'result.json',dict(candidates=summaries,deployed=False,final_not_used=True));print(json.dumps(summary),flush=True)
            if gate:return
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash

if __name__=='__main__':main()
