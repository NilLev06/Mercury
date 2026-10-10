"""Matched full-history neural-teacher fitting, frozen versus slow live core."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint,save_checkpoint
from fly_brain_rc.imitation.curriculum import validation_cases
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from scripts.train_live_core import load_live_source,fit_live_sequences
from scripts.train_precontact_dagger import expanded_gate
from scripts.train_neural_ppo import freeze_actor
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','replay','baseline-report','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--scope',choices=['frozen','live'],required=True)
    parser.add_argument('--replay-source',type=Path)
    parser.add_argument('--resume-optimizer',action='store_true')
    parser.add_argument('--stage-offset',type=int,default=0)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'run.py').write_text(Path(__file__).read_text());(args.out/'fit.snapshot.py').write_text(Path('scripts/train_live_core.py').read_text())
    raw=torch.load(args.source,weights_only=True);model=load_live_source(args.source,raw['graph'])
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest();saved=torch.load(args.replay,weights_only=True)
    replay_source=args.replay_source or args.source
    if saved['source_sha256']!=hashlib.sha256(replay_source.read_bytes()).hexdigest():raise ValueError('teacher source/replay mismatch')
    teacher_raw=torch.load(replay_source,weights_only=True)
    if raw['signature']!=teacher_raw['signature'] or raw['schema']!=teacher_raw['schema']:raise ValueError('source/teacher graph or schema mismatch')
    if args.stage_offset<0:raise ValueError('nonnegative stage offset required')
    if not saved['collection_complete']:raise ValueError('incomplete replay')
    replay=saved['episodes'];anchors=[(e['x'],e['y']) for e in replay if e['teaching_actor']=='old_source']
    if not anchors:raise ValueError('missing actual source anchors')
    if args.scope=='frozen':
        freeze_actor(model);model.trainable_core=True # existing sequence fitter accepts this flag; gradients remain frozen
    protected={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad};mask=model.mask.clone()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    if args.resume_optimizer:
        optimizer.load_state_dict(raw['optimizer'])
        if not optimizer.state:raise ValueError('empty optimizer continuation')
    expanded=[v for seed in range(930001,930005) for v in mixed_cases(seed,'selection',2)]
    baseline=json.loads(args.baseline_report.read_text());results=[]
    write(args.out/'protocol.json',dict(source_sha256=source_hash,replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        scope=args.scope,replay_source=str(replay_source.resolve()),resume_optimizer=args.resume_optimizer,stage_offset=args.stage_offset,learning_rate=1e-5,anchor_weight=4.,opening_weight=1.,stages=3,updates_per_stage=200,
        fitting_seeds=[970001+args.stage_offset+i for i in range(3)],selection_seeds=list(range(930001,930005)),reserved_final_seeds=list(range(950001,950005))))
    for local_stage in range(3):
        stage=local_stage+args.stage_offset
        torch.manual_seed(970001+stage);model.trainable_core=True
        loss=fit_live_sequences(model,optimizer,replay,anchors,updates=200,anchor_weight=4.,opening_weight=1.)
        assert torch.equal(model.mask,mask) and torch.all(model.effective_weights()[mask==0]==0)
        assert all(torch.equal(dict(model.named_parameters())[n],p) for n,p in protected.items())
        model.trainable_core=args.scope=='live'
        model.training_metadata['matched_neural_scope']=dict(scope=args.scope,source_sha256=source_hash,
            learning_rate=1e-5,anchor_weight=4.,updates=200*(stage+1),seed=970001+stage,full_history=True,topology_verified=True)
        save_checkpoint(args.out/f'candidate-{stage}.pt',model,optimizer,[],raw['iteration'])
        old=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);previous=evaluate(model,mixed_cases(20001,'selection',1));new=evaluate(model,expanded)
        gate=expanded_gate(old,previous,new,baseline)
        summary=dict(scope=args.scope,stage=stage,updates=200*(stage+1),loss=loss,old=old['successes'],old_contacts=old['contacts'],
            previous_complex=sum(r['family']=='complex' and r['success'] and r['contacts']==0 for r in previous['reports']),
            expanded={tier:dict(clean=sum(r['success'] and r['contacts']==0 for r in new['reports'] if r['family']==tier),contacts=sum(r['contacts'] for r in new['reports'] if r['family']==tier)) for tier in ('simple','complex')},development_gate=gate)
        write(args.out/f'candidate-{stage}.json',dict(summary=summary,old=old,previous=previous,expanded=new));results.append(summary)
        write(args.out/'result.json',dict(candidates=results,deployed=False,final_not_used=True));print(json.dumps(summary),flush=True)
        if gate:break
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash

if __name__=='__main__':main()
