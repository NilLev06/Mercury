"""Teacher-labelled DN-only motor fitting; frozen anatomy, resumable replay.

Collection/evaluation reuse the existing imitation runner. This module does
not enable a new dashboard mode or promote an experimental checkpoint.
"""
from pathlib import Path
import torch

from .model import graph_signature
from .sensorimotor_policy import SensorimotorPolicy,observable_drive

SCHEMA='sensorimotor-dn-only-cv-area-v2'


def _validate_optimizer(model,optimizer):
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError('resumable trials support Adam only')
    parameters={id(p) for group in optimizer.param_groups for p in group['params']}
    if parameters!={id(p) for p in model.head.parameters()}:
        raise ValueError('optimizer may update only the motor head')


def cache_demonstrations(model,trajectories,*,batch_size=8):
    if (not trajectories or type(batch_size) is not int or batch_size<1
            or any(x.ndim!=2 or x.shape[1]!=19 or y.shape!=(len(x),2) or not len(x)
                   or not torch.isfinite(x).all() or not torch.isfinite(y).all()
                   or (y.abs()>1).any() for x,y in trajectories)):
        raise ValueError('finite aligned nonempty observable/teacher sequences required')
    cache=[]
    with torch.no_grad():
        for start in range(0,len(trajectories),batch_size):
            group=trajectories[start:start+batch_size]
            x=torch.nn.utils.rnn.pad_sequence([v[0] for v in group],batch_first=True)
            state=None;features=[]
            for frame in x.unbind(1):
                drive=observable_drive(frame,model.core.channels)
                for _ in range(5):state=model.core.step(drive,state)
                features.append(model.core.readout(state).float())
            dn=torch.stack(features,1)
            for i,(values,labels) in enumerate(group):
                cache.append((dn[i,:len(values)].clone(),labels.clone()))
    return cache


def fit_cached(model,optimizer,cache,*,updates=1000,batch_size=128):
    if (not cache or type(updates) is not int or not 1<=updates<=10000
            or type(batch_size) is not int or not 1<=batch_size<=4096):
        raise ValueError('nonempty cache and bounded fitting settings required')
    _validate_optimizer(model,optimizer)
    for z,y in cache:
        if (z.ndim!=2 or z.shape[1]!=len(model.graph['outputs']) or not len(z)
                or y.shape!=(len(z),2) or not torch.isfinite(z).all()
                or not torch.isfinite(y).all() or (y.abs()>1).any()):
            raise ValueError('finite aligned DN/teacher cache required')
    model.train();losses=[]
    # Uniform episodes rather than uniform frames: a stalled long rollout
    # must not outweigh many successful short demonstrations.
    for _ in range(updates):
        chosen=torch.randint(len(cache),(batch_size,))
        features=[];labels=[]
        for episode in chosen.tolist():
            z,y=cache[episode];index=int(torch.randint(len(z),()))
            features.append(z[index]);labels.append(y[index])
        z=torch.stack(features);y=torch.stack(labels)
        z=(z-model.feature_mean)/model.feature_scale.clamp_min(1e-8)
        prediction=torch.tanh(model.head(z))
        loss=(prediction-y).square().mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
        optimizer.step();losses.append(float(loss.detach()))
    return sum(losses[-min(100,len(losses)):])/min(100,len(losses))


def save_trial(path,model,optimizer,trajectories,*,rounds):
    _validate_optimizer(model,optimizer)
    payload=dict(schema=SCHEMA,graph=model.graph,signature=graph_signature(model.graph),
                 model=model.state_dict(),optimizer=optimizer.state_dict(),
                 trajectories=trajectories,rounds=rounds,rng=torch.get_rng_state(),
                 vehicle_integration_allowed=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:torch.save(payload,handle)


def load_trial(path):
    payload=torch.load(path,weights_only=True,map_location='cpu')
    if (payload.get('schema')!=SCHEMA
            or payload.get('signature')!=graph_signature(payload['graph'])
            or type(payload.get('rounds')) is not int or payload['rounds']<0):
        raise ValueError('matching sensorimotor trial schema and graph required')
    model=SensorimotorPolicy(payload['graph'])
    expected=model.core.state_dict()
    for name,values in expected.items():
        actual=payload['model']['core.'+name]
        if values.is_sparse:
            values,actual=values.coalesce(),actual.coalesce()
            equal=(torch.equal(values.indices(),actual.indices())
                   and torch.equal(values.values(),actual.values()))
        else:equal=torch.equal(values,actual)
        if not equal:raise ValueError('frozen anatomy/dynamics mismatch')
    model.load_state_dict(payload['model'])
    optimizer=torch.optim.Adam(model.head.parameters(),lr=.003)
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng'])
    return model,optimizer,payload['trajectories'],payload['rounds']
