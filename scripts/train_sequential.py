"""Progressive neural avoidance training with real baseline rollout anchors.

No live checkpoint writes. Output directory must be new. All teaching occurs
in collection; selection and final evaluation are autonomous.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.imitation.curriculum import place_start, validation_cases
from fly_brain_rc.imitation.model import load_checkpoint, save_checkpoint
from fly_brain_rc.imitation.navigation_cases import sequential_cases, navigation_cases
from fly_brain_rc.imitation.runner import collect_episode
from fly_brain_rc.imitation.validation import validate
from fly_brain_rc.sim.env import FlyCarEnv
from scripts.improve_navigation import neural_states, build_optimizer, write, rank


def episode(model, case, pose, *, beta=None, steps=2400,teacher_factory=None):
    observations = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: observations.append(inputs[0][0, 0].detach().clone()))
    env = None
    try:
        env = FlyCarEnv(SIM, DRIVE, scenario=case, render=True)
        place_start(env, pose)
        trajectory, report = collect_episode(env, model, beta=beta, steps=steps,teacher_factory=teacher_factory)
        x = torch.stack(observations)
        report.update(scenario=case.name, start_pose=pose, beta=beta)
        print(json.dumps(dict(phase='episode', **report)), flush=True)
        return x, trajectory[1] if trajectory else None, report
    finally:
        hook.remove()
        if env is not None and env.renderer is not None:
            env.renderer.close()


def evaluate(model, cases):
    reports = []
    for case, pose, family in cases:
        _, _, report = episode(model, case, pose)
        report['family'] = family
        reports.append(report)
    return dict(episodes=len(reports), successes=sum(r['success'] for r in reports),
                contacts=sum(r['contacts'] for r in reports), reports=reports)


def categories(x, y):
    contact = x[:,18] > .5
    result = torch.where(contact, 2, torch.where(y[:,1].abs() > .3, 1, 0))
    result[:min(10,len(x))] = 3
    for i in torch.where(contact)[0].tolist():
        result[max(10,i-10):i] = 4
    result[y[:,0].abs() < .02] = 5
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    payload = torch.load(args.source, weights_only=True)
    model, _, old, age = load_checkpoint(args.source, payload['graph'])
    if model.motor_head != 'residual32':
        raise ValueError('residual32 source required')
    torch.set_num_threads(2); torch.manual_seed(80261008)
    base = {k:v.clone() for k,v in model.state_dict().items()}
    frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    optimizer = build_optimizer(model, 3e-5)
    # Preserve the actual state distribution of the successful deployed model,
    # rather than relying only on old teacher trajectory observations.
    live_h, live_y, live_x, fixed_reports = [], [], [], []
    for case in validation_cases():
        x, _, report = episode(model, case, (0.,0.,0.))
        h, y = neural_states(model, x)
        live_x.append(x); live_h.append(h); live_y.append(y); fixed_reports.append(report)
    if sum(r['success'] for r in fixed_reports) != 7 or sum(r['contacts'] for r in fixed_reports):
        raise RuntimeError('Baseline no longer meets seven-map gate')
    live_h, live_y = torch.cat(live_h), torch.cat(live_y)
    old_h, old_y = [], []
    for x, _ in old:
        h, y = neural_states(model, x); safe = x[:,18] < .5
        old_h.append(h[safe]); old_y.append(y[safe])
    old_h, old_y = torch.cat(old_h), torch.cat(old_y)
    selection_cases = sequential_cases(7201,'selection',count=8,obstacles=2)
    selection_cases += [v for v in navigation_cases(7202,'selection') if v[2]=='double']
    baseline_selection = evaluate(model, selection_cases)
    write(args.out/'baseline-selection.json', baseline_selection)
    torch.save(dict(observations=live_x, states=live_h, outputs=live_y,
                    reports=fixed_reports, source_sha256=source_hash), args.out/'live-anchors.pt')
    replay, collection, candidates = [], [], []
    updates_since_reset, reset_cycles = 0, []
    for cycle, count in enumerate((1,2,3)):
        for case, pose, family in sequential_cases(8201+cycle,'train',count=8,obstacles=count):
            for beta in (1.,0.):
                try:
                    x,y,r = episode(model,case,pose,beta=beta,steps=2400 if beta==1 else 1600)
                    h,_ = neural_states(model,x)
                    replay.append((x,y,h,categories(x,y)))
                    r.update(cycle=cycle,teacher_demo=beta==1.);collection.append(r)
                except ValueError as error:
                    r=dict(scenario=case.name,cycle=cycle,beta=beta,error=str(error))
                    collection.append(r);print(json.dumps(r),flush=True)
        write(args.out/'collection.json',collection)
        h,y,c = torch.cat([v[2] for v in replay]),torch.cat([v[1] for v in replay]),torch.cat([v[3] for v in replay])
        replay_path=args.out/f'replay-{cycle}.pt'
        torch.save(dict(trajectories=[(v[0],v[1]) for v in replay],states=h,labels=y,categories=c,
                        live_anchor_states=live_h,live_anchor_outputs=live_y,
                        old_anchor_states=old_h,old_anchor_outputs=old_y),replay_path)
        pools=[torch.where(c==i)[0] for i in range(6)];pools=[v for v in pools if len(v)]
        for step in range(1000):
            index=torch.cat([pool[torch.randint(len(pool),(48,))] for pool in pools])
            a=torch.randint(len(live_h),(384,));b=torch.randint(len(old_h),(128,))
            prediction=torch.tanh(model.readout(h[index]))
            current=torch.tanh(model.readout(live_h[a]));historic=torch.tanh(model.readout(old_h[b]))
            w=torch.tensor([1.,2.])
            loss=((prediction-y[index]).square()*w).mean()
            loss+=32*((current-live_y[a]).square()*w).mean()
            loss+=4*((historic-old_y[b]).square()*w).mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout.residual.parameters(),1.)
            optimizer.step()
        assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
        model.training_metadata['sequential_experiment']=dict(
            source_sha256=source_hash,source_iteration=age,train_seeds=list(range(8201,8202+cycle)),
            selection_seeds=[7201,7202],final_seeds=[9201,9202],total_experiment_updates=1000*(cycle+1),
            updates_since_reset=updates_since_reset, reset_after_cycles=list(reset_cycles),
            replay=str(replay_path.resolve()),live_anchors=str((args.out/'live-anchors.pt').resolve()),
            collected_cases=collection,learning_rate=3e-5,live_anchor_weight=32,old_anchor_weight=4)
        path=args.out/f'candidate-{cycle}.pt';save_checkpoint(path,model,optimizer,old,age)
        fixed=validate(model);selection=evaluate(model,selection_cases)
        report=dict(fixed=fixed,new=selection,loss=float(loss.detach()))
        write(args.out/f'candidate-{cycle}-selection.json',report);candidates.append((path,report))
        print(json.dumps(dict(phase='candidate',cycle=cycle,fixed=fixed['successes'],fixed_contacts=fixed['contacts'],
                             new=selection['successes'],new_contacts=selection['contacts'])),flush=True)
        # A regression does not become the foundation for the next curriculum
        # stage. Replay still accumulates; weights and optimizer return to base.
        if fixed['successes']!=7 or fixed['contacts']:
            model.load_state_dict(base);optimizer=build_optimizer(model,3e-5)
            reset_cycles.append(cycle); updates_since_reset=0
    eligible=[v for v in candidates if v[1]['fixed']['successes']==7 and v[1]['fixed']['contacts']==0
              and not v[1]['fixed']['cancelled'] and v[1]['new']['successes']>baseline_selection['successes']]
    if eligible:
        path,report=max(eligible,key=lambda v:rank(v[1]['new']))
        model,_,_,_=load_checkpoint(path,payload['graph'])
        final_cases=sequential_cases(9201,'final',count=12,obstacles=2)+sequential_cases(9202,'final',count=12,obstacles=3)
        candidate_final=evaluate(model,final_cases)
        model.load_state_dict(base);baseline_final=evaluate(model,final_cases)
        result=dict(selected=str(path),candidate_final=candidate_final,baseline_final=baseline_final,
                    accepted=candidate_final['successes']>baseline_final['successes'] and candidate_final['contacts']<=baseline_final['contacts'])
    else:
        result=dict(selected=None,accepted=False,reason='No transfer candidate retains original seven-map gate')
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==source_hash
    result.update(source_sha256=source_hash,deployed=False,frozen_core_verified=True)
    write(args.out/'result.json',result)
    print(json.dumps(dict(phase='complete',selected=result['selected'],accepted=result['accepted'])),flush=True)


if __name__=='__main__':
    main()
