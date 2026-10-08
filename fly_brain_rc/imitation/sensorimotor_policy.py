"""Observable CV/range/touch adapter -> frozen graph -> DN-only motor head.

Mappings are explicit engineering hypotheses, not measured fly responses.
No coordinates, compass, planner, or raw-observation bypass reaches the head.
"""
import torch
from torch import nn

from .observations import INPUT_SIZE
from .sensorimotor_model import SensorimotorCore


def observable_drive(x,channels):
    if x.ndim!=2 or x.shape[1]!=INPUT_SIZE or not torch.isfinite(x).all():
        raise ValueError('finite batch x 19 observable features required')
    columns={}
    for side,direction,diagonal,lateral,contact in (
            ('left',-1,7,9,16),('right',1,8,10,17)):
        target=(x[:,0].clamp(0,1)*(1+direction*x[:,1].clamp(-1,1))/2
                *(.25+.75*x[:,2].clamp(0,1)))
        obstacle=x[:,3].clamp(0,1)*(1+direction*x[:,4].clamp(-1,1))/2
        # No claim of natural looming: these are observable proximity proxies.
        near=1-torch.stack((x[:,6],x[:,diagonal],x[:,lateral]),1).clamp(0,1).amin(1)
        columns['LC10a:'+side]=target
        columns['LC4:'+side]=obstacle*x[:,5].clamp(0,1)
        columns['LPLC2:'+side]=torch.maximum(obstacle*x[:,5].clamp(0,1),near)
        columns['BM_InOm:'+side]=torch.maximum(x[:,contact],x[:,14]).clamp(0,1)
    if not channels or any(name not in columns for name in channels):
        raise ValueError('known annotated population channels required')
    return torch.stack([columns[name] for name in channels],1)*.3


class SensorimotorPolicy(nn.Module):
    def __init__(self,graph,*,learned_input=False,motor_memory=False):
        super().__init__()
        self.graph=graph
        self.core=SensorimotorCore(graph)
        self.learned_input=learned_input
        self.motor_memory=motor_memory
        if learned_input:
            # A vehicle-to-population interface, NOT biological tuning curves.
            # No observation is discarded before the learned projection.
            self.encoder=nn.Sequential(nn.Linear(INPUT_SIZE,16),nn.Tanh(),
                                       nn.Linear(16,len(self.core.channels)))
            nn.init.zeros_(self.encoder[-1].weight)
            nn.init.zeros_(self.encoder[-1].bias)
        self.head=nn.Sequential(nn.Linear(len(graph['outputs']),16),nn.Tanh(),nn.Linear(16,2))
        if motor_memory:
            # Explicit external DN-history memory, NOT anatomical recurrence.
            self.memory=nn.GRUCell(len(graph['outputs']),16)
            self.memory_readout=nn.Linear(16,2)
            nn.init.zeros_(self.memory_readout.weight)
            nn.init.zeros_(self.memory_readout.bias)
            with torch.no_grad():self.memory.bias_hh[16:32].fill_(2.2)
        self.register_buffer('feature_mean',torch.zeros(len(graph['outputs'])))
        self.register_buffer('feature_scale',torch.ones(len(graph['outputs'])))

    def input_drive(self,frame):
        base=observable_drive(frame,self.core.channels)
        return (base+.05*torch.tanh(self.encoder(frame))) if self.learned_input else base

    def motor_step(self,dn,memory=None):
        features=(dn-self.feature_mean)/self.feature_scale.clamp_min(1e-8)
        output=self.head(features)
        if self.motor_memory:
            if memory is None:memory=features.new_zeros(len(features),16)
            memory=self.memory(features,memory)
            output=output+self.memory_readout(memory)
        return torch.tanh(output),memory

    def forward(self,x,state=None):
        if x.ndim!=3 or x.shape[1]<1:
            raise ValueError('nonempty batch x time x observation sequence required')
        outputs=[];memory=None
        if self.motor_memory and state is not None:
            if state.shape!=(len(x),self.core.neurons+16) or not torch.isfinite(state).all():
                raise ValueError('aligned finite anatomical and motor memory state required')
            state,memory=state[:,:self.core.neurons],state[:,self.core.neurons:].float()
        for frame in x.unbind(1):
            drive=self.input_drive(frame)
            # Five neural substeps per camera decision are a model choice;
            # frozen dynamics have no trainable physiological interpretation.
            # Gradients may cross fixed anatomy to calibrate the sensory
            # interface, but no core buffer becomes a trainable parameter.
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.learned_input):
                for _ in range(5):state=self.core.step(drive,state)
                dn=self.core.readout(state).float()
            output,memory=self.motor_step(dn,memory)
            outputs.append(output)
        if self.motor_memory:state=torch.cat((state,memory.to(state.dtype)),1)
        return torch.stack(outputs,1),state
