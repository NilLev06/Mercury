"""Teacher-labelled DN-only motor fitting; frozen anatomy, resumable replay.

Collection/evaluation reuse the existing imitation runner. This module does
not enable a new dashboard mode or promote an experimental checkpoint.
"""
from pathlib import Path
import torch

from .model import graph_signature
from .sensorimotor_policy import SensorimotorPolicy

SCHEMA='sensorimotor-dn-only-cv-area-v2'
INTERFACE_SCHEMA='sensorimotor-learned-input-dn-only-v1'


def _validate_optimizer(model,optimizer):
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError('resumable trials support Adam only')
    parameters={id(p) for group in optimizer.param_groups for p in group['params']}
    ordered=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(ordered)!=len(parameters) or parameters!={id(p) for p in model.parameters()}:
        raise ValueError('optimizer must contain exactly the trainable interfaces')


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
                drive=model.input_drive(frame)
                for _ in range(5):state=model.core.step(drive,state)
                features.append(model.core.readout(state).float())
            dn=torch.stack(features,1)
            for i,(values,labels) in enumerate(group):
                cache.append((dn[i,:len(values)].clone(),labels.clone()))
    return cache


def fit_cached(model,optimizer,cache,*,updates=1000,batch_size=128):
    if model.learned_input:
        raise ValueError('learned inputs require live recurrent fitting, not stale cached features')
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


def fit_interfaces(model,optimizer,trajectories,*,updates=100,batch_size=4,
                   sequence_length=8):
    """Fit sensory/motor interfaces through immutable anatomical propagation.

    Equal episode sampling; 24-frame detached burn-in; padded labels masked.
    No reward changes, teacher coordinates, or direct motor bypass.
    """
    if (not model.learned_input or not trajectories
            or type(updates) is not int or not 1<=updates<=10000
            or type(batch_size) is not int or not 1<=batch_size<=64
            or type(sequence_length) is not int or not 1<=sequence_length<=48
            or any(x.ndim!=2 or x.shape[1]!=19 or y.shape!=(len(x),2) or not len(x)
                   or not torch.isfinite(x).all() or not torch.isfinite(y).all()
                   or (y.abs()>1).any() for x,y in trajectories)):
        raise ValueError('learned interfaces and finite aligned bounded fitting settings required')
    _validate_optimizer(model,optimizer)
    model.train();losses=[]
    for _ in range(updates):
        inputs=[];labels=[];states=[]
        for episode in torch.randint(len(trajectories),(batch_size,)).tolist():
            x,y=trajectories[episode];length=min(len(x),sequence_length)
            start=int(torch.randint(len(x)-length+1,()))
            with torch.no_grad():
                state=(model(x[max(0,start-24):start][None])[1] if start else
                       model.core.drive.new_zeros(1,model.core.neurons))
            inputs.append(x[start:start+length]);labels.append(y[start:start+length]);states.append(state)
        lengths=torch.tensor([len(x) for x in inputs])
        x=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence(labels,batch_first=True)
        predicted,_=model(x,torch.cat(states))
        valid=torch.arange(x.shape[1])[None,:]<lengths[:,None]
        loss=((predicted-y).square().mean(-1)*valid).sum()/valid.sum()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();losses.append(float(loss.detach()))
    return sum(losses[-min(100,len(losses)):])/min(100,len(losses))


def save_trial(path,model,optimizer,trajectories,*,rounds):
    _validate_optimizer(model,optimizer)
    names={id(p):name for name,p in model.named_parameters()}
    payload=dict(schema=INTERFACE_SCHEMA if model.learned_input else SCHEMA,
                 graph=model.graph,signature=graph_signature(model.graph),
                 model=model.state_dict(),optimizer=optimizer.state_dict(),
                 optimizer_names=[[names[id(p)] for p in group['params']]
                                  for group in optimizer.param_groups],
                 trajectories=trajectories,rounds=rounds,rng=torch.get_rng_state(),
                 vehicle_integration_allowed=False)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:torch.save(payload,handle)


def load_trial(path):
    payload=torch.load(path,weights_only=True,map_location='cpu')
    if (payload.get('schema') not in (SCHEMA,INTERFACE_SCHEMA)
            or payload.get('signature')!=graph_signature(payload['graph'])
            or type(payload.get('rounds')) is not int or payload['rounds']<0):
        raise ValueError('matching sensorimotor trial schema and graph required')
    model=SensorimotorPolicy(payload['graph'],learned_input=payload['schema']==INTERFACE_SCHEMA)
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
    names=payload.get('optimizer_names')
    parameters=dict(model.named_parameters())
    if names is None:
        # Original head-only schema had a single canonical group.
        if model.learned_input:
            raise ValueError('learned interface checkpoint requires named optimizer groups')
        names=[list(parameters)]
    flattened=[name for group in names for name in group]
    if (len(flattened)!=len(set(flattened)) or set(flattened)!=set(parameters)
            or len(names)!=len(payload['optimizer']['param_groups'])
            or any(len(group)!=len(saved['params']) for group,saved in
                   zip(names,payload['optimizer']['param_groups']))):
        raise ValueError('aligned named optimizer groups required')
    optimizer=torch.optim.Adam([{'params':[parameters[name] for name in group]}
                                for group in names],lr=.003)
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng'])
    return model,optimizer,payload['trajectories'],payload['rounds']
