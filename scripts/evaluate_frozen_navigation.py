"""Evaluate one frozen candidate and original baseline on predeclared new maps."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fly_brain_rc.imitation.model import load_checkpoint
from fly_brain_rc.imitation.navigation_cases import mixed_cases
from fly_brain_rc.imitation.curriculum import validation_cases
from scripts.train_sequential import evaluate
from scripts.improve_navigation import write


def frozen_gate(old,candidate,baseline,protocol):
    metric=protocol.get('simple_metric','successes')
    if metric not in ('successes','clean_successes'):raise ValueError('unknown simple success metric')
    return (old['successes']==7 and old['contacts']==0
        and candidate['complex']['successes']>=protocol['minimum_complex_successes']
        and candidate['complex']['contacts']==0
        and candidate['simple'][metric]>=baseline['simple'][metric]
        and candidate['simple']['contacts']<=baseline['simple']['contacts'])


def main():
    folder=Path(sys.argv[1]);protocol=json.loads((folder/'protocol.json').read_text());torch.set_num_threads(2)
    candidate=folder/'selected.pt';baseline=Path(protocol['baseline'])
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=protocol['candidate_sha256']:raise ValueError('frozen candidate changed')
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=protocol['baseline_sha256']:raise ValueError('baseline changed')
    cases=[c for seed in protocol['final_seeds'] for c in mixed_cases(seed,'final',2)]
    report={}
    for name,path in [('candidate',candidate),('baseline',baseline)]:
        payload=torch.load(path,weights_only=True);model,*_=load_checkpoint(path,payload['graph'])
        if name=='candidate':report['candidate_old']=evaluate(model,[(c,(0.,0.,0.),'old') for c in validation_cases()]);write(folder/'partial.json',report)
        report[name]=evaluate(model,cases);write(folder/'partial.json',report)
    breakdown={}
    for name in ('candidate','baseline'):
        breakdown[name]={}
        for tier in ('simple','complex'):
            r=[v for v in report[name]['reports'] if v['family']==tier]
            breakdown[name][tier]=dict(episodes=len(r),successes=sum(v['success'] for v in r),clean_successes=sum(v['success'] and v['contacts']==0 for v in r),contacts=sum(v['contacts'] for v in r))
    old=report['candidate_old'];c=breakdown['candidate'];b=breakdown['baseline']
    accepted=frozen_gate(old,c,b,protocol)
    result=dict(protocol=protocol,breakdown=breakdown,reports=report,accepted=accepted,deployed=False,
        frozen_candidate_verified=hashlib.sha256(candidate.read_bytes()).hexdigest()==protocol['candidate_sha256'],
        baseline_unchanged=hashlib.sha256(baseline.read_bytes()).hexdigest()==protocol['baseline_sha256'])
    write(folder/'result.json',result);print(json.dumps(dict(phase='frozen_final',accepted=accepted,breakdown=breakdown)),flush=True)


if __name__=='__main__':main()
