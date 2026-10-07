"""Offline type-level calibration for stationary cue-conditioned memory.

Measured counts remain fixed. Target moments describe an ideal test direction,
NOT recorded fly activity; this is not a replication of a self-supervised
velocity-integration paper and does not train the deployed vehicle.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .anatomical_heading import AnatomicalHeading, expanded_parameters, extend_phases
from .heading_columns import digest


class TypeParameterFit(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.recurrence=network.recurrence
        self.kinds=tuple(sorted(set(network._parameter_types)))
        index={kind:i for i,kind in enumerate(self.kinds)}
        p=network.parameters
        gains=torch.tensor([[p['gain'][post][pre] for pre in self.kinds]
                            for post in self.kinds],dtype=torch.float32)
        if torch.any(gains==0):
            raise ValueError('fit requires explicit nonzero sign hypotheses')
        self.register_buffer('counts',torch.tensor(network.synapse_counts,dtype=torch.float32))
        self.register_buffer('sign',gains.sign())
        self.register_buffer('type_indices',torch.tensor([index[k] for k in network._parameter_types]))
        self.register_buffer('tau',torch.tensor(network._tau,dtype=torch.float32))
        self.register_buffer('cue_indices',torch.tensor(network._cue_indices))
        self.register_buffer('cue_phases',torch.tensor(network._cue_phases,dtype=torch.float32))
        groups=np.zeros((len(network.root_ids),len(network._groups)))
        for j,indices in enumerate(network._groups):
            groups[indices,j]=1/len(indices)
        self.register_buffer('groups',torch.tensor(groups,dtype=torch.float32))
        angles=torch.tensor(np.angle(network._directions),dtype=torch.float32)
        self.register_buffer('moments',torch.stack((angles.cos(),angles.sin()),dim=1)/len(angles))
        self.log_gain=nn.Parameter(gains.abs().log())
        self.bias=nn.Parameter(torch.tensor([p['bias'][k] for k in self.kinds]))
        self.register_buffer('initial_log_gain',self.log_gain.detach().clone())
        self.register_buffer('initial_bias',self.bias.detach().clone())
        self._base_parameters=p

    def weights(self):
        gains=self.sign*self.log_gain.clamp(math.log(.01),math.log(50)).exp()
        return self.counts/self._base_parameters['synapse_scale']*gains[
            self.type_indices[:,None],self.type_indices[None,:]]

    def _cue(self, phases):
        cue=torch.zeros(len(phases),len(self.counts),dtype=phases.dtype,device=phases.device)
        cue[:,self.cue_indices]=self._base_parameters['cue_strength']*torch.cos(
            self.cue_phases[None,:]-phases[:,None])
        return cue

    def _advance(self,state,dt,weights,cue):
        steps=max(1,math.ceil(dt/min(.01,float(self.tau.min())/5)))
        h=dt/steps
        bias=self.bias.clamp(-10,10)[self.type_indices]
        for _ in range(steps):
            feedback=state@weights.T if self.recurrence else 0.
            target=torch.sigmoid((feedback+bias+cue).clamp(-60,60))
            state=state+h/self.tau*(target-state)
        return state

    def step(self,state,dt,*,cue_phase=None):
        if not math.isfinite(dt) or not 0<dt<=1:
            raise ValueError('0<dt<=1 required')
        if state.ndim!=2 or state.shape[1]!=len(self.counts) or not torch.isfinite(state).all():
            raise ValueError('finite batch x neuron state required')
        cue=0.
        if cue_phase is not None:
            if cue_phase.shape!=(len(state),) or not torch.isfinite(cue_phase).all():
                raise ValueError('finite cue phase per batch element required')
            cue=self._cue(cue_phase)
        return self._advance(state,dt,self.weights(),cue)

    def equilibrium_residual(self,state):
        """Dark target-rate minus activity; zero is necessary, not sufficient,
        for stable memory. Used only for offline feasibility diagnostics.
        """
        if (state.ndim!=2 or len(state)==0 or state.shape[1]!=len(self.counts)
                or not torch.isfinite(state).all() or torch.any((state<0)|(state>1))):
            raise ValueError('finite bounded batch x neuron activity required')
        feedback=state@self.weights().T if self.recurrence else 0.
        drive=feedback+self.bias.clamp(-10,10)[self.type_indices]
        return torch.sigmoid(drive.clamp(-60,60))-state

    def profile_loss(self,state,phases):
        """Ideal centered cosine profile, not a physiological recording.

        Unlike first-moment loss, this detects higher-harmonic distortions.
        Neither this target nor a fitted auxiliary state is injected at runtime.
        """
        self.equilibrium_residual(state)  # Same activity-domain validation.
        if phases.shape!=(len(state),) or not torch.isfinite(phases).all():
            raise ValueError('finite cue phase per batch element required')
        angles=torch.atan2(self.moments[:,1],self.moments[:,0])
        desired=.3*torch.cos(angles[None,:]-phases[:,None])
        desired=desired-desired.mean(dim=1,keepdim=True)
        rates=state@self.groups
        return ((rates-rates.mean(dim=1,keepdim=True)-desired)**2).mean()

    def memory_loss(self,phases,*,cue_steps=50,dark_steps=200):
        if (phases.ndim!=1 or len(phases)==0 or not torch.isfinite(phases).all()
                or cue_steps<1 or dark_steps<1):
            raise ValueError('finite phases and positive simulation lengths required')
        state=torch.zeros(len(phases),len(self.counts),dtype=phases.dtype,device=phases.device)
        weights=self.weights();cue=self._cue(phases)
        for _ in range(cue_steps):
            state=self._advance(state,.01,weights,cue)
        # Ideal cue supplies targets. No simulated heading is injected in darkness.
        desired=.15*torch.stack((phases.cos(),phases.sin()),dim=1)
        losses=[]
        for i in range(dark_steps):
            state=self._advance(state,.01,weights,0.)
            if (i+1)%10==0 or i==dark_steps-1:
                rates=state@self.groups
                rates=rates-rates.mean(dim=1,keepdim=True)
                losses.append(100*((rates@self.moments-desired)**2).mean())
        regularizer=.001*((self.log_gain-self.initial_log_gain)**2).mean()
        regularizer+=.001*((self.bias-self.initial_bias)**2).mean()
        return torch.stack(losses).mean()+regularizer

    def parameters_dict(self):
        gains=(self.sign*self.log_gain.clamp(math.log(.01),math.log(50)).exp()).detach().cpu().numpy()
        bias=self.bias.clamp(-10,10).detach().cpu().numpy()
        p=copy.deepcopy(self._base_parameters)
        p['gain']={post:{pre:float(gains[i,j]) for j,pre in enumerate(self.kinds)}
                   for i,post in enumerate(self.kinds)}
        p['bias']={k:float(bias[i]) for i,k in enumerate(self.kinds)}
        return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy',type=Path,required=True)
    parser.add_argument('--phases',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    if args.out.exists() or not 1<=args.steps<=10000:
        parser.error('new output path and 1..10000 steps required')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    anatomy=json.loads(args.anatomy.read_text())
    phases=extend_phases(anatomy,json.loads(args.phases.read_text()))
    model=AnatomicalHeading(anatomy,phases,expanded_parameters(anatomy))
    trial=TypeParameterFit(model)
    optimizer=torch.optim.Adam(trial.parameters(),lr=.03)
    directions=torch.linspace(-math.pi,math.pi,13)[:-1]+.173
    history=[]
    for step in range(args.steps):
        optimizer.zero_grad()
        loss=trial.memory_loss(directions)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite memory calibration loss; no checkpoint written')
        loss.backward();torch.nn.utils.clip_grad_norm_(trial.parameters(),5.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step%20==0 or step==args.steps-1:
            print(json.dumps(dict(step=step+1,loss=history[-1])),flush=True)
    checkpoint=dict(version=1,task='stationary ideal-cue memory calibration',
                    steps=args.steps,seed=args.seed,parameters=trial.parameters_dict(),
                    state_dict=trial.state_dict(),optimizer=optimizer.state_dict(),loss_history=history,
                    anatomy_sha256=digest(args.anatomy),phases_sha256=digest(args.phases),
                    root_ids=model.root_ids,vehicle_integration_allowed=False,
                    limitations=['Fixed sign hypotheses include simplified effective neuromodulation.',
                                 'Ideal direction targets, not recorded fly activity.',
                                 'No fitted time constants, yaw integration or deployed car training.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive file creation avoids even a concurrent overwrite of an older run.
    with args.out.open('xb') as stream:
        torch.save(checkpoint,stream)


if __name__=='__main__':
    main()
