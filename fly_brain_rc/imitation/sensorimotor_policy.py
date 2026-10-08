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
    def __init__(self,graph):
        super().__init__()
        self.graph=graph
        self.core=SensorimotorCore(graph)
        self.head=nn.Sequential(nn.Linear(len(graph['outputs']),16),nn.Tanh(),nn.Linear(16,2))
        self.register_buffer('feature_mean',torch.zeros(len(graph['outputs'])))
        self.register_buffer('feature_scale',torch.ones(len(graph['outputs'])))

    def forward(self,x,state=None):
        if x.ndim!=3 or x.shape[1]<1:
            raise ValueError('nonempty batch x time x observation sequence required')
        outputs=[]
        for frame in x.unbind(1):
            drive=observable_drive(frame,self.core.channels)
            # Five neural substeps per camera decision are a model choice;
            # frozen dynamics have no trainable physiological interpretation.
            with torch.no_grad():
                for _ in range(5):state=self.core.step(drive,state)
                dn=self.core.readout(state).float()
            features=(dn-self.feature_mean)/self.feature_scale.clamp_min(1e-8)
            outputs.append(torch.tanh(self.head(features)))
        return torch.stack(outputs,1),state
