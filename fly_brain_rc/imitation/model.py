"""Trainable anatomical recurrence, arbitrary learned sensory/motor adapters."""
from pathlib import Path
import hashlib
import json
import copy
import torch
from torch import nn
from .observations import SCHEMA,SECTOR_SCHEMA,SCHEMA_SIZES


def graph_signature(graph):
    return hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest()


class ResidualMotorHead(nn.Module):
    """Learned motor adapter; no planner, map or hidden target inputs."""
    def __init__(self,base):
        super().__init__()
        self.base = base
        self.residual = nn.Sequential(nn.Linear(base.in_features,32),nn.Tanh(),nn.Linear(32,2))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self,state):
        return self.base(state)+self.residual(state)


class RecurrentPolicy(nn.Module):
    def __init__(self,graph,baseline=False,*,motor_head='linear',trainable_core=False,
                 observation_schema=SCHEMA):
        super().__init__()
        if motor_head not in ('linear','residual32'):
            raise ValueError('unknown motor head architecture')
        if type(trainable_core) is not bool:
            raise ValueError('trainable_core must be boolean')
        self.trainable_core = trainable_core
        if observation_schema not in SCHEMA_SIZES:
            raise ValueError('unknown observation schema')
        self.observation_schema = observation_schema
        self.motor_head = motor_head
        self.graph,self.baseline = graph,baseline
        self.training_metadata = {'seeds':[],'cases':[]}
        n = len(graph['ids'])
        counts = torch.zeros(n,n)
        for pre,post,count in graph['edges']:
            counts[post,pre] = count
        mask = torch.ones_like(counts) if baseline else (counts>0).float()
        self.register_buffer('mask',mask)
        initial = torch.randn(n,n)*(.1/n**.5) if baseline else torch.log1p(counts)
        if not baseline:
            initial = initial/initial.sum(dim=1,keepdim=True).clamp_min(1)*.5
        self.recurrent = nn.Parameter(initial)
        self.encoder = nn.Linear(SCHEMA_SIZES[observation_schema],n)
        self.bias = nn.Parameter(torch.zeros(n))
        self.leak = nn.Parameter(torch.zeros(n))
        self.readout = nn.Linear(n,2)
        if motor_head=='residual32':
            self.readout = ResidualMotorHead(self.readout)
            # Default residual training preserves sensory/recurrent dynamics.
            # Explicit experimental full-core training enables existing weights.
            for name,parameter in self.named_parameters():
                parameter.requires_grad_(trainable_core or 'readout.residual.' in name)

    def effective_weights(self):
        return self.recurrent*self.mask

    def forward(self,x,state=None):
        if state is None:
            state = x.new_zeros(x.shape[0],self.mask.shape[0])
        outputs = []
        alpha = torch.sigmoid(self.leak)
        weights = self.effective_weights()
        for values in x.unbind(1):
            candidate = torch.tanh(self.encoder(values)+state@weights.T+self.bias)
            state = (1-alpha)*state+alpha*candidate
            outputs.append(torch.tanh(self.readout(state)))
        return torch.stack(outputs,dim=1),state


def upgrade_sector_inputs(source):
    if source.observation_schema != SCHEMA:
        raise ValueError('v1 source required for sector upgrade')
    upgraded = RecurrentPolicy(source.graph,source.baseline,motor_head=source.motor_head,
                              trainable_core=True,observation_schema=SECTOR_SCHEMA)
    state = {key:value.clone() for key,value in source.state_dict().items()}
    state['encoder.weight'] = torch.nn.functional.pad(state['encoder.weight'],(0,12))
    upgraded.load_state_dict(state)
    upgraded.training_metadata = copy.deepcopy(source.training_metadata)
    return upgraded


def save_checkpoint(path,model,optimizer,trajectories,iteration):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = {'version':1,'schema':model.observation_schema,'graph':model.graph,
               'motor_head':model.motor_head,'trainable_core':model.trainable_core,
               'signature':graph_signature(model.graph),'baseline':model.baseline,
               'model':model.state_dict(),'optimizer':optimizer.state_dict(),
               'training_metadata':model.training_metadata,
               'trajectories':trajectories,'iteration':iteration,'torch_rng':torch.get_rng_state()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def load_checkpoint(path,graph,*,baseline=False):
    # weights_only avoids executing arbitrary pickle globals in model files.
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if (payload.get('version')!=1 or payload.get('schema') not in SCHEMA_SIZES
            or payload.get('signature')!=graph_signature(graph)
            or payload.get('baseline')!=baseline):
        raise ValueError('checkpoint graph, model or observation schema mismatch')
    model = RecurrentPolicy(graph,baseline,motor_head=payload.get('motor_head','linear'),
                            trainable_core=payload.get('trainable_core',False),
                            observation_schema=payload['schema'])
    model.load_state_dict(payload['model'])
    model.training_metadata = payload.get('training_metadata',{'seeds':[],'cases':[]})
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng'])
    return model,optimizer,payload['trajectories'],payload['iteration']
