"""Autonomous model selection; validation never collects or trains."""
import json
import math
from pathlib import Path
from ..config import SIM,DRIVE
from ..sim.env import FlyCarEnv
from .curriculum import validation_cases
from .model import graph_signature,save_checkpoint

SUITE = 'navigation-seven-v1'


def validate(model,*,steps=2400,cancel=None,emit=None):
    from .runner import collect_episode
    reports = []
    for case in validation_cases():
        if cancel and cancel():
            break
        env = FlyCarEnv(SIM,DRIVE,seed=0,scenario=case,render=True)
        try:
            _,report = collect_episode(env,model,steps=min(2400,steps),beta=None,cancel=cancel,emit=emit)
            report['scenario'] = case.name
            reports.append(report)
            if report['cancelled']:
                break
        finally:
            if env.renderer is not None:
                env.renderer.close()
    return dict(suite=SUITE,step_limit=min(2400,steps),episodes=len(reports),
                successes=sum(r['success'] for r in reports),
                contacts=sum(r['contacts'] for r in reports),
                mean_final_distance=sum(r['final_goal_distance'] for r in reports)/max(1,len(reports)),
                cancelled=len(reports)!=7 or any(r['cancelled'] for r in reports),reports=reports)


def best_paths(path):
    path = Path(path)
    return path.with_name(path.stem+'.best.pt'),path.with_name(path.stem+'.validation.json')


def previous_validation(path,model,steps=2400):
    best,metadata = best_paths(path)
    if not best.exists() or not metadata.exists():
        return None
    try:
        result = json.loads(metadata.read_text())
        if (result['signature']!=graph_signature(model.graph) or result['baseline']!=model.baseline
                or result['suite']!=SUITE or result['step_limit']!=min(2400,steps)):
            return None
        return result
    except (OSError,ValueError,KeyError):
        return None


def promote_best(path,model,optimizer,data,iteration,result):
    if result['cancelled'] or result['episodes']!=7:
        return False
    def rank(r):
        return r['successes'],-r['contacts'],-r['mean_final_distance']
    if not math.isfinite(result['mean_final_distance']):
        return False
    steps = result.get('step_limit',2400)
    previous = previous_validation(path,model,steps)
    if previous is not None and rank(result)<=rank(previous):
        return False
    best,metadata = best_paths(path)
    save_checkpoint(best,model,optimizer,data,iteration)
    payload = dict(result,suite=SUITE,step_limit=steps,iteration=iteration,
                   signature=graph_signature(model.graph),baseline=model.baseline)
    temporary = metadata.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temporary.replace(metadata)
    return True
