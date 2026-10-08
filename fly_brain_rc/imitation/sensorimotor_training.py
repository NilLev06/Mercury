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
MEMORY_SCHEMA='sensorimotor-dn-history-memory-v1'


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
    if model.learned_input or model.motor_memory:
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


def fit_memory(model,optimizer,cache,*,updates=100,batch_size=8,sequence_length=32):
    """Learn external DN-history memory, with complete detached prefix replay.

    Only fixed-core features are cached. Every crop rebuilds motor memory
    from the episode start with current parameters, never stale hidden states.
    """
    if (not model.motor_memory or model.learned_input or not cache
            or type(updates) is not int or not 1<=updates<=10000
            or type(batch_size) is not int or not 1<=batch_size<=64
            or type(sequence_length) is not int or not 1<=sequence_length<=128
            or any(z.ndim!=2 or z.shape[1]!=len(model.graph['outputs']) or not len(z)
                   or y.shape!=(len(z),2) or not torch.isfinite(z).all()
                   or not torch.isfinite(y).all() or (y.abs()>1).any() for z,y in cache)):
        raise ValueError('fixed sensory interface, motor memory and aligned bounded cache required')
    _validate_optimizer(model,optimizer)
    model.train();losses=[]
    for _ in range(updates):
        inputs=[];labels=[];prefixes=[]
        for episode in torch.randint(len(cache),(batch_size,)).tolist():
            z,y=cache[episode];length=min(len(z),sequence_length)
            start=int(torch.randint(len(z)-length+1,()))
            prefixes.append(z[:start]);inputs.append(z[start:start+length]);labels.append(y[start:start+length])
        memory=inputs[0].new_zeros(batch_size,16)
        if any(len(z) for z in prefixes):
            prefix_lengths=torch.tensor([len(z) for z in prefixes])
            prefix=torch.nn.utils.rnn.pad_sequence(prefixes,batch_first=True)
            with torch.no_grad():
                for i,frame in enumerate(prefix.unbind(1)):
                    _,candidate=model.motor_step(frame,memory)
                    memory=torch.where((i<prefix_lengths)[:,None],candidate,memory)
        lengths=torch.tensor([len(z) for z in inputs])
        z=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True)
        y=torch.nn.utils.rnn.pad_sequence(labels,batch_first=True)
        predicted=[]
        for frame in z.unbind(1):
            output,memory=model.motor_step(frame,memory);predicted.append(output)
        predicted=torch.stack(predicted,1)
        valid=torch.arange(z.shape[1])[None,:]<lengths[:,None]
        loss=((predicted-y).square().mean(-1)*valid).sum()/valid.sum()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();losses.append(float(loss.detach()))
    return sum(losses[-min(100,len(losses)):])/min(100,len(losses))


def fit_interfaces(model,optimizer,trajectories,*,updates=100,batch_size=4,
                   sequence_length=8):
    """Fit sensory/motor interfaces through immutable anatomical propagation.

    Equal episode sampling; detached burn-in; padded labels masked. Motor
    memory reconstructs the full prefix, not just the core's 24-frame context.
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
                state=(model(x[0 if model.motor_memory else max(0,start-24):start][None])[1] if start else
                       model.core.drive.new_zeros(1,model.core.neurons+(16 if model.motor_memory else 0)))
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
    payload=dict(schema=MEMORY_SCHEMA if model.motor_memory else
                 INTERFACE_SCHEMA if model.learned_input else SCHEMA,
                 learned_input=model.learned_input,
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
    if (payload.get('schema') not in (SCHEMA,INTERFACE_SCHEMA,MEMORY_SCHEMA)
            or payload.get('signature')!=graph_signature(payload['graph'])
            or type(payload.get('rounds')) is not int or payload['rounds']<0):
        raise ValueError('matching sensorimotor trial schema and graph required')
    memory=payload['schema']==MEMORY_SCHEMA
    if memory and type(payload.get('learned_input')) is not bool:
        raise ValueError('explicit sensory interface required for memory checkpoint')
    model=SensorimotorPolicy(payload['graph'],motor_memory=memory,
                            learned_input=payload['learned_input'] if memory else
                            payload['schema']==INTERFACE_SCHEMA)
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
        if model.learned_input or model.motor_memory:
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
