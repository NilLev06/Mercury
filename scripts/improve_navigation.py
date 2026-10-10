"""Isolated residual-only experiment; never installs or overwrites live weights.

Run with the flybrain Python. MuJoCo camera rendering needs macOS graphics
access. Output must be a new directory. Final seeds are evaluated only after
candidate selection, and are never used to choose weights or hyperparameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.imitation.model import load_checkpoint, save_checkpoint
from fly_brain_rc.imitation.runner import collect_episode
from fly_brain_rc.imitation.curriculum import place_start
from fly_brain_rc.imitation.navigation_cases import navigation_cases
from fly_brain_rc.imitation.validation import validate
from fly_brain_rc.sim.env import FlyCarEnv


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def assess(model, seeds, split, families):
    reports = []
    for seed in seeds:
        for case, pose, family in navigation_cases(seed, split):
            if family not in families:
                continue
            env = FlyCarEnv(SIM, DRIVE, scenario=case, render=True)
            try:
                place_start(env, pose)
                _, report = collect_episode(env, model, steps=2400)
                report.update(scenario=case.name, family=family, start_pose=pose)
                reports.append(report)
                print(json.dumps(dict(phase='evaluate', **report)), flush=True)
            finally:
                env.renderer.close()
    return dict(episodes=len(reports), successes=sum(r['success'] for r in reports),
                contacts=sum(r['contacts'] for r in reports), reports=reports)


def neural_states(model, x):
    states = []
    hook = model.readout.register_forward_pre_hook(
        lambda module, inputs: states.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad():
            outputs, _ = model(x[None])
    finally:
        hook.remove()
    return torch.cat(states), outputs[0].detach()


def build_optimizer(model, learning_rate):
    # The standard loader reconstructs Adam over all parameters. Frozen
    # parameters receive no gradients, but must stay in its parameter group.
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def blend_full_policy(base,candidate,scale):
    if not 0<=scale<=1 or base.keys()!=candidate.keys():raise ValueError('bounded matched policy states required')
    if 'mask' not in base or not torch.equal(base['mask'],candidate['mask']):raise ValueError('anatomical mask differs')
    if any(base[k].shape!=candidate[k].shape for k in base):raise ValueError('policy shapes differ')
    if scale==0:return {k:v.clone() for k,v in base.items()}
    if scale==1:return {k:v.clone() for k,v in candidate.items()}
    return {k:v.clone() if k=='mask' else v+scale*(candidate[k]-v) for k,v in base.items()}


def blend_residual(base, candidate, scale):
    if not 0 <= scale <= 1 or base.keys() != candidate.keys():
        raise ValueError('aligned checkpoint states and bounded scale required')
    mixed = {}
    for name, value in base.items():
        if name.startswith('readout.residual.'):
            mixed[name] = value + scale*(candidate[name]-value)
        else:
            if not torch.equal(value, candidate[name]):
                raise ValueError('candidate changed frozen core or base readout')
            mixed[name] = value.clone()
    return mixed


def rank(report):
    return report['successes'], -report['contacts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--families', nargs='+', choices=('double','staggered','search'), default=['double','staggered','search'])
    parser.add_argument('--teacher-demos', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--anchor-weight', type=float, default=4.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    payload = torch.load(args.source, map_location='cpu', weights_only=True)
    model, _, old, iteration = load_checkpoint(args.source, payload['graph'])
    if model.motor_head != 'residual32':
        raise ValueError('user-selected residual32 checkpoint required')
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    baseline_state = {name: value.clone() for name, value in model.state_dict().items()}
    torch.manual_seed(20261008)
    optimizer = build_optimizer(model, args.lr)
    # Entire episode histories, rather than short recurrent burn-in, provide
    # exact cached states because all state-producing parameters are frozen.
    anchors_h, anchors_y = [], []
    for x, _ in old:
        h, y = neural_states(model, x)
        safe = x[:, 18] < .5
        anchors_h.append(h[safe]); anchors_y.append(y[safe])
    anchors_h, anchors_y = torch.cat(anchors_h), torch.cat(anchors_y)
    baseline_fixed = validate(model)
    baseline_selection = assess(model, (4101, 4102), 'selection', args.families)
    write(args.out/'baseline-selection.json', dict(fixed=baseline_fixed, new=baseline_selection))
    cache_h, cache_y, categories, collection = [], [], [], []
    candidates = []
    for cycle, seed in enumerate((5101, 5102)):
        for case, pose, family in navigation_cases(seed, 'train'):
            if family not in args.families:
                continue
            env = FlyCarEnv(SIM, DRIVE, scenario=case, render=True)
            try:
                place_start(env, pose)
                if args.teacher_demos:
                    demo, demo_report = collect_episode(env, model, steps=2400, beta=1.)
                    dx, dy = demo
                    dh, _ = neural_states(model, dx)
                    dc = torch.where(dy[:, 1].abs() > .3, 1, 0)
                    dc[:min(10, len(dx))] = 3
                    cache_h.append(dh); cache_y.append(dy); categories.append(dc)
                    demo_report.update(scenario=case.name, family=family, cycle=cycle, teacher_demo=True)
                    collection.append(demo_report)
                    print(json.dumps(dict(phase='demonstrate', **demo_report)), flush=True)
                    env.reset(); place_start(env, pose)
                blocked = [0]
                def cancel():
                    blocked[0] = blocked[0]+1 if env._contact_active else 0
                    return blocked[0] >= 100
                trajectory, report = collect_episode(env, model, steps=1600, beta=0., cancel=cancel)
                x, y = trajectory
                h, _ = neural_states(model, x)
                contact = x[:, 18] > .5
                category = torch.where(contact, 2, torch.where(y[:, 1].abs() > .3, 1, 0))
                category[:min(10, len(x))] = 3
                for i in torch.where(contact)[0].tolist():
                    category[max(10, i-10):i] = 4
                cache_h.append(h); cache_y.append(y); categories.append(category)
                report.update(scenario=case.name, family=family, cycle=cycle)
                collection.append(report)
                print(json.dumps(dict(phase='collect', **report)), flush=True)
            except ValueError as error:
                collection.append(dict(scenario=case.name, cycle=cycle, error=str(error)))
                print(json.dumps(collection[-1]), flush=True)
            finally:
                env.renderer.close()
        write(args.out/'collection.json', collection)
        if not cache_h:
            continue
        h, y, category = torch.cat(cache_h), torch.cat(cache_y), torch.cat(categories)
        pools = [torch.where(category == i)[0] for i in range(5)]
        pools = [pool for pool in pools if len(pool)]
        for update in range(600):
            indices = torch.cat([pool[torch.randint(len(pool), (64,))] for pool in pools])
            anchor = torch.randint(len(anchors_h), (256,))
            prediction = torch.tanh(model.readout(h[indices]))
            retained = torch.tanh(model.readout(anchors_h[anchor]))
            weights = torch.tensor([1., 2.])
            loss = ((prediction-y[indices]).square()*weights).mean()
            loss += args.anchor_weight*((retained-anchors_y[anchor]).square()*weights).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.readout.residual.parameters(), 1.)
            optimizer.step()
        assert all(torch.equal(dict(model.named_parameters())[name], value)
                   for name, value in frozen.items()), 'Frozen core changed'
        replay_path = args.out/f'cached-training-{cycle}.pt'
        torch.save(dict(states=h, labels=y, categories=category,
                        anchor_states=anchors_h, anchor_outputs=anchors_y), replay_path)
        model.training_metadata['navigation_experiment'] = dict(
            source_sha256=source_hash, training_seeds=[5101, 5102][:cycle+1],
            selection_seeds=[4101, 4102], final_seeds=[6101, 6102],
            families=args.families, teacher_demos=args.teacher_demos,
            learning_rate=args.lr, anchor_weight=args.anchor_weight,
            updates=600*(cycle+1), collected_cases=collection,
            cached_replay=str(replay_path.resolve()),
            source_iteration=iteration,
            replay_note='Checkpoint trajectories retain source anchor episodes; experimental replay is the separate cached-state artifact.')
        path = args.out/f'candidate-{cycle}.pt'
        save_checkpoint(path, model, optimizer, old, iteration)
        fixed = validate(model)
        selection = assess(model, (4101, 4102), 'selection', args.families)
        report = dict(fixed=fixed, new=selection, loss=float(loss.detach()))
        write(args.out/f'candidate-{cycle}-selection.json', report)
        candidates.append((path, report))
    eligible = [(path, r) for path, r in candidates
                if r['fixed']['successes'] == 7 and r['fixed']['contacts'] == 0
                and not r['fixed']['cancelled']
                and rank(r['new']) > rank(baseline_selection)
                and r['new']['successes'] > 0]
    if eligible:
        selected, selection_report = max(eligible, key=lambda item: rank(item[1]['new']))
        model, _, _, _ = load_checkpoint(selected, payload['graph'])
        candidate_final = assess(model, (6101, 6102), 'final', args.families)
        model.load_state_dict(baseline_state)
        baseline_final = assess(model, (6101, 6102), 'final', args.families)
        # More successes must not conceal additional contact incidents.
        accepted = (rank(candidate_final) > rank(baseline_final)
                    and candidate_final['successes'] >= baseline_final['successes']
                    and candidate_final['contacts'] <= baseline_final['contacts'])
        result = dict(selected=str(selected), accepted=accepted,
                      baseline_final=baseline_final, candidate_final=candidate_final)
    else:
        result = dict(selected=None, accepted=False,
                      reason='No candidate beats selection baseline while retaining seven-map gate')
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_hash
    result.update(source_sha256=source_hash, training_seeds=[5101, 5102],
                  selection_seeds=[4101, 4102], final_seeds=[6101, 6102],
                  frozen_parameters_verified=True, deployed=False, families=args.families,
                  teacher_demos=args.teacher_demos, learning_rate=args.lr, anchor_weight=args.anchor_weight)
    write(args.out/'result.json', result)
    print(json.dumps(dict(phase='complete', accepted=result['accepted'], selected=result['selected'])), flush=True)


if __name__ == '__main__':
    main()
