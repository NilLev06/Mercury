"""Frozen sparse anatomical signal pilot; dynamics are engineering choices.

Transmitter signs are coarse proxies, not receptor-specific physiology.
Per-row absolute normalization and tanh/leak are declared model assumptions.
Only designated input neurons receive drive; only designated DNs are read.
"""
import math
import torch
from torch import nn


class SensorimotorCore(nn.Module):
    def __init__(self,graph,*,gain=.8,leak=.25):
        super().__init__()
        if not math.isfinite(gain) or not 0<gain<=1 or not math.isfinite(leak) or not 0<leak<=1:
            raise ValueError('bounded finite gain/leak required')
        n=len(graph['ids'])
        valid_index=lambda i:type(i) is int and 0<=i<n
        inputs=graph['inputs'];outputs=graph['outputs']
        if (n<1 or len(set(graph['ids']))!=n or len(graph['neurons'])!=n
                or not inputs or any(not group for group in inputs.values())
                or not outputs or len(set(outputs))!=len(outputs)
                or any(not valid_index(i) for group in inputs.values() for i in group)
                or any(not valid_index(i) for i in outputs)
                or set(outputs)&{i for group in inputs.values() for i in group}):
            raise ValueError('aligned annotated graph with disjoint bounded input/output indices required')
        if any('flow' in graph['neurons'][i] and graph['neurons'][i]['flow']!='efferent'
               for i in outputs):
            raise ValueError('annotated outputs must be efferent neurons')
        if any(len(edge)!=3 or not valid_index(edge[0]) or not valid_index(edge[1])
               or not isinstance(edge[2],(int,float)) or not math.isfinite(edge[2])
               or edge[2]<=0 or int(edge[2])!=edge[2] for edge in graph['edges']):
            raise ValueError('bounded anatomical edges with positive integer counts required')
        self.channels=tuple(sorted(graph['inputs']))
        self.neurons=n;self.leak=leak
        pre,post,counts=zip(*graph['edges']) if graph['edges'] else ((),(),())
        pre=torch.tensor(pre,dtype=torch.long);post=torch.tensor(post,dtype=torch.long)
        counts=torch.tensor(counts,dtype=torch.float64)
        sign=torch.tensor([-1. if r.get('top_nt','').lower() in
            ('gaba','glutamate','histamine') else 1. for r in graph['neurons']],dtype=torch.float64)
        total=torch.zeros(n,dtype=torch.float64).scatter_add_(0,post,counts)
        weights=counts/total[post].clamp_min(1)*sign[pre]*gain
        self.register_buffer('weights',torch.sparse_coo_tensor(
            torch.stack((post,pre)),weights,(n,n),check_invariants=True).coalesce())
        drive=torch.zeros(len(self.channels),n,dtype=torch.float64)
        for i,name in enumerate(self.channels):drive[i,graph['inputs'][name]]=1.
        self.register_buffer('drive',drive)
        self.register_buffer('output_indices',torch.tensor(graph['outputs'],dtype=torch.long))

    def step(self,x,state=None):
        if x.ndim!=2 or x.shape[1]!=len(self.channels) or not torch.isfinite(x).all():
            raise ValueError('finite batch x input-channel values required')
        x=x.to(dtype=self.drive.dtype,device=self.drive.device)
        if state is None:state=x.new_zeros(len(x),self.neurons)
        if state.shape!=(len(x),self.neurons) or not torch.isfinite(state).all():
            raise ValueError('finite aligned neural state required')
        feedback=torch.sparse.mm(self.weights,state.T).T
        return (1-self.leak)*state+self.leak*torch.tanh(feedback+x@self.drive)

    def readout(self,state):
        return state[:,self.output_indices]
