"""Independent held-out synthetic channel recovery, NOT navigation evaluation."""
import argparse
import json
from pathlib import Path

import torch

from .connectome import checksum
from .sensorimotor_model import SensorimotorCore


def evaluate_channels(graph,*,samples=128,steps=64,seed=82):
    if type(samples) is not int or not 16<=samples<=1024 or type(steps) is not int or not 3<=steps<=256:
        raise ValueError('bounded trial settings required')
    rng=torch.Generator().manual_seed(seed)
    channels=tuple(sorted(graph['inputs']))
    x=torch.rand(2*samples,len(channels),generator=rng,dtype=torch.float64)*.3
    result={}
    for name in ('intact','disconnected','shuffled_labels'):
        selected=dict(graph)
        if name=='disconnected':selected['edges']=[]
        core=SensorimotorCore(selected)
        with torch.no_grad():
            state=None
            for _ in range(steps):state=core.step(x,state)
            features=core.readout(state)
            mean=features[:samples].mean(0)
            std=features[:samples].std(0).clamp_min(1e-14)
            features=(features-mean)/std
            features=torch.cat((features,torch.ones(len(x),1,dtype=x.dtype)),dim=1)
            labels=x[:samples]
            if name=='shuffled_labels':labels=labels[torch.randperm(samples,generator=rng)]
            weights=torch.linalg.lstsq(features[:samples],labels,driver='gelsd').solution
            prediction=features[samples:]@weights
            mse=(prediction-x[samples:]).square().mean(0)
            r2=1-mse/x[samples:].var(0,unbiased=False)
            result[name]={channel:dict(r2=float(r2[i]),rmse=float(mse[i].sqrt()))
                          for i,channel in enumerate(channels)}
    return dict(**result,samples_train=samples,samples_test=samples,steps=steps,seed=seed,
                task='synthetic-input-amplitude-recovery',
                assumptions='frozen signed row-normalized tanh rate network; engineered population drive',
                vehicle_integration_allowed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('new output path required')
    torch.set_num_threads(1)
    result=evaluate_channels(json.loads(args.graph.read_text()))
    result['graph_sha256']=checksum(args.graph)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
