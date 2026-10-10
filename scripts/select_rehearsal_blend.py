"""Finite development-only static blends of frozen-core neural motor heads."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.improve_navigation import blend_residual,blend_full_policy,write
from scripts.train_sequential import evaluate
from scripts.train_precontact_dagger import expanded_gate
from scripts.train_neural_ppo import freeze_actor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','toward','baseline-report','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--full-policy',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'run.py').write_text(Path(__file__).read_text())
    raw=torch.load(args.base,weights_only=True);other=torch.load(args.toward,weights_only=True);model,*_=load_checkpoint(args.base,raw['graph']);freeze_actor(model)
    if any(raw[k]!=other[k] for k in ('signature','schema','motor_head')):raise ValueError('model architecture or graph differs')
    if args.full_policy:model.trainable_core=True
    baseline=json.loads(args.baseline_report.read_text());expanded_cases=[v for seed in range(930001,930005) for v in mixed_cases(seed,'selection',2)]
    protocol=dict(base=str(args.base.resolve()),base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
        toward=str(args.toward.resolve()),toward_sha256=hashlib.sha256(args.toward.read_bytes()).hexdigest(),scales=[.25,.5,.75],full_policy=args.full_policy,
        baseline_report=str(args.baseline_report.resolve()),baseline_report_sha256=hashlib.sha256(args.baseline_report.read_bytes()).hexdigest(),
        selection_seeds=list(range(930001,930005)),reserved_final_seeds=list(range(950001,950005)))
    write(args.out/'protocol.json',protocol);results=[]
    for scale in protocol['scales']:
        model.load_state_dict((blend_full_policy if args.full_policy else blend_residual)(raw['model'],other['model'],scale))
        old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);previous=None;expanded=None;gate=False
        if old['successes']==7 and old['contacts']==0:
            previous=evaluate(model,mixed_cases(20001,'selection',1))
            c=[r for r in previous['reports'] if r['family']=='complex']
            if all(r['success'] and r['contacts']==0 for r in c):
                expanded=evaluate(model,expanded_cases);gate=expanded_gate(old,previous,expanded,baseline)
        model.training_metadata['rehearsal_parameter_blend']=dict(protocol,scale=scale,optimizer_reset=True,runtime='single neural network')
        save_checkpoint(args.out/f'candidate-{scale}.pt',model,torch.optim.Adam(model.parameters(),lr=3e-4),[],raw['iteration'])
        summary=dict(scale=scale,old=old['successes'],old_contacts=old['contacts'],previous_complex=None if previous is None else sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in previous['reports']),
            expanded=None if expanded is None else {tier:dict(clean=sum(r['success'] and r['contacts']==0 for r in expanded['reports'] if r['family']==tier),contacts=sum(r['contacts'] for r in expanded['reports'] if r['family']==tier)) for tier in ('simple','complex')},development_gate=gate)
        write(args.out/f'candidate-{scale}.json',dict(summary=summary,old=old,previous=previous,expanded=expanded));results.append(summary)
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True));print(json.dumps(summary),flush=True)
        if gate:break

if __name__=='__main__':main()
