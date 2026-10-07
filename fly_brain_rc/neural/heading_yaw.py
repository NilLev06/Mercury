"""Isolated GLNO input hypothesis; no scalar heading integrator at runtime.

Velocity stimulates annotated GLNO neurons only. Positive-side convention and
input gain must be calibrated; no claim of validated biological transduction.
"""
import math
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .heading_type_fit import TypeParameterFit
from .anatomical_heading import AnatomicalHeading,extend_phases
from .heading_columns import digest


def glno_masks(network,positive_side):
    if positive_side not in ('left','right'):
        raise ValueError('positive side must be left or right')
    classes=network.parameters.get('cell_classes',{})
    masks={side:np.zeros(len(network.root_ids)) for side in ('left','right')}
    for i,(root,kind) in enumerate(zip(network.root_ids,network._types)):
        if kind!='GLNO':continue
        label=classes.get(root)
        if label not in ('GLNO:left','GLNO:right'):
            raise ValueError('explicit GLNO side annotations required')
        masks[label.split(':')[1]][i]=1.
    if not all(mask.any() for mask in masks.values()):
        raise ValueError('both annotated GLNO sides required')
    other='right' if positive_side=='left' else 'left'
    return masks[positive_side],masks[other]


def glno_drive(network,yaw_rate):
    """NumPy replay of the saved sensory mapping, not an angle/state update."""
    config=network.parameters.get('yaw_input',{})
    gain=config.get('gain');side=config.get('positive_side')
    if (not math.isfinite(yaw_rate) or abs(yaw_rate)>5 or gain is None
            or not math.isfinite(gain) or not .01<=gain<=20):
        raise ValueError('bounded finite yaw and saved GLNO input gain required')
    positive,negative=glno_masks(network,side)
    if config.get('mode','rectified')=='differential':
        return gain*yaw_rate*(positive-negative)
    if config.get('mode','rectified')!='rectified':
        raise ValueError('unknown GLNO input mode')
    return gain*(max(yaw_rate,0)*positive+max(-yaw_rate,0)*negative)


class GLNOYawFit(TypeParameterFit):
    def __init__(self,network,*,input_gain=None,positive_side=None,input_mode=None,learn_tau=False):
        super().__init__(network,learn_tau=learn_tau)
        saved=network.parameters.get('yaw_input',{})
        input_gain=saved.get('gain',2.) if input_gain is None else input_gain
        positive_side=saved.get('positive_side','left') if positive_side is None else positive_side
        input_mode=saved.get('mode','rectified') if input_mode is None else input_mode
        if input_mode not in ('rectified','differential'):
            raise ValueError('unknown GLNO input mode')
        if not math.isfinite(input_gain) or not .01<=input_gain<=20:
            raise ValueError('input gain must be in .01..20')
        positive,negative=glno_masks(network,positive_side)
        self.register_buffer('positive_glno',torch.tensor(positive,dtype=torch.float32))
        self.register_buffer('negative_glno',torch.tensor(negative,dtype=torch.float32))
        self.log_input_gain=nn.Parameter(torch.tensor(math.log(input_gain)))
        self.register_buffer('positive_side_code',torch.tensor(0 if positive_side=='left' else 1))
        self.register_buffer('input_mode_code',torch.tensor(0 if input_mode=='rectified' else 1))

    @property
    def positive_side(self):
        return ('left','right')[int(self.positive_side_code)]

    @property
    def input_mode(self):
        return ('rectified','differential')[int(self.input_mode_code)]

    def drive(self,velocities):
        if (velocities.ndim!=1 or not torch.isfinite(velocities).all()
                or torch.any(velocities.abs()>5)):
            raise ValueError('finite yaw batch bounded by 5 rad/s required')
        gain=self.log_input_gain.clamp(math.log(.01),math.log(20)).exp()
        if self.input_mode=='differential':
            return gain*velocities[:,None]*(self.positive_glno-self.negative_glno)
        return gain*(velocities.clamp_min(0)[:,None]*self.positive_glno
                     +(-velocities).clamp_min(0)[:,None]*self.negative_glno)

    def step_yaw(self,state,dt,velocities,*,cue_phase=None):
        return self.step(state,dt,cue_phase=cue_phase,external_drive=self.drive(velocities))

    def velocity_loss(self,phases,velocities,*,cue_steps=50,drive_steps=200,speed_weight=.02):
        """Offline supervised ideal rotation task, NOT the paper's exact loss.

        The reference angle exists only in this training objective. Runtime gets
        angular velocity and neural state, never the integrated reference angle.
        """
        if (phases.ndim!=1 or len(phases)==0 or phases.shape!=velocities.shape
                or not torch.isfinite(phases).all() or not isinstance(cue_steps,int)
                or not isinstance(drive_steps,int) or cue_steps<1 or drive_steps<1
                or not math.isfinite(speed_weight) or not 0<=speed_weight<=100):
            raise ValueError('matching finite phase/velocity batches and positive integer lengths required')
        drive=self.drive(velocities)
        state=torch.zeros(len(phases),len(self.counts),dtype=phases.dtype,device=phases.device)
        weights=self.weights();cue=self._cue(phases)
        for _ in range(cue_steps):state=self._advance(state,.01,weights,cue)
        rates=state@self.groups
        previous=(rates-rates.mean(dim=1,keepdim=True))@self.moments
        losses=[];last_sample=0
        for i in range(drive_steps):
            state=self._advance(state,.01,weights,drive)
            if (i+1)%10==0 or i==drive_steps-1:
                angle=phases+velocities*((i+1)*.01)
                desired=.15*torch.stack((angle.cos(),angle.sin()),dim=1)
                rates=state@self.groups
                moment=(rates-rates.mean(dim=1,keepdim=True))@self.moments
                cross=previous[:,0]*moment[:,1]-previous[:,1]*moment[:,0]
                dot=(previous*moment).sum(dim=1)
                speed=torch.atan2(cross,dot+1e-8)/((i+1-last_sample)*.01)
                losses.append(100*(moment-desired).square().mean()
                              +speed_weight*(speed-velocities).square().mean())
                previous=moment;last_sample=i+1
        regularizer=.001*((self.log_gain-self.initial_log_gain).square().mean()
                          +(self.bias-self.initial_bias).square().mean())
        return torch.stack(losses).mean()+regularizer

    def parameters_dict(self):
        params=super().parameters_dict()
        gain=float(self.log_input_gain.detach().clamp(math.log(.01),math.log(20)).exp())
        params['yaw_input']=dict(gain=max(.01,min(20.,gain)),positive_side=self.positive_side,
            mode=self.input_mode,units='rad/s',mapping='drive to annotated GLNO sides',
            validated_physiology=False)
        return params


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('anatomy','phases','checkpoint','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--learn-tau',action='store_true')
    parser.add_argument('--input-mode',choices=['rectified','differential'])
    parser.add_argument('--lr',type=float)
    parser.add_argument('--speed-weight',type=float,default=2.)
    args=parser.parse_args(argv)
    if (args.out.exists() or not 1<=args.steps<=10000
            or args.lr is not None and (not math.isfinite(args.lr) or not 0<args.lr<=.1)
            or not math.isfinite(args.speed_weight) or not 0<=args.speed_weight<=100):
        parser.error('new output, bounded steps, finite valid learning settings required')
    checkpoint=torch.load(args.checkpoint,weights_only=True)
    if (checkpoint.get('anatomy_sha256')!=digest(args.anatomy)
            or checkpoint.get('phases_sha256')!=digest(args.phases)):
        raise ValueError('checkpoint input provenance does not match')
    anatomy=json.loads(args.anatomy.read_text())
    phases=extend_phases(anatomy,json.loads(args.phases.read_text()))
    network=AnatomicalHeading(anatomy,phases,checkpoint['parameters'])
    if tuple(checkpoint.get('root_ids',()))!=network.root_ids:
        raise ValueError('checkpoint neuron order does not match')
    learn_tau=args.learn_tau or (args.resume and checkpoint.get('learn_tau',
                                  'log_tau' in checkpoint.get('state_dict',{})))
    trial=GLNOYawFit(network,learn_tau=learn_tau,input_mode=args.input_mode)
    optimizer=torch.optim.Adam(trial.parameters(),lr=args.lr or .003)
    completed=int(checkpoint.get('yaw_training_steps',0))
    if args.resume:
        state=checkpoint['state_dict']
        dynamics=dict(recurrence=trial.recurrence,
                      synapse_scale=network.parameters['synapse_scale'],
                      cue_strength=network.parameters['cue_strength'])
        if checkpoint.get('dynamics_context')!=dynamics:
            raise ValueError('checkpoint scalar dynamics context does not match')
        context=('counts','sign','type_indices','cue_indices','cue_phases','groups','moments',
                 'positive_glno','negative_glno','positive_side_code','input_mode_code')
        if not learn_tau:context=context+('tau',)
        if any(name not in state or not torch.equal(state[name],getattr(trial,name)) for name in context):
            raise ValueError('checkpoint fixed context does not match')
        trial.load_state_dict(state)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.lr is not None:
            for group in optimizer.param_groups:group['lr']=args.lr
    torch.set_num_threads(1)
    history=[]
    for i in range(args.steps):
        phase=(torch.tensor([-2.,0.,2.])+(completed+i)*.381966).repeat_interleave(7)
        velocity=torch.tensor([-.8,-.4,-.15,0.,.15,.4,.8]).repeat(3)
        optimizer.zero_grad()
        loss=trial.velocity_loss(phase,velocity,speed_weight=args.speed_weight)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite yaw loss; checkpoint not written')
        loss.backward();torch.nn.utils.clip_grad_norm_(trial.parameters(),5.);optimizer.step()
        history.append(float(loss.detach()))
        if i%50==0 or i==args.steps-1:
            print(json.dumps(dict(yaw_step=completed+i+1,loss=history[-1])),flush=True)
    result=dict(version=1,task='offline ideal-cue GLNO angular integration calibration',
                steps=completed+args.steps,yaw_training_steps=completed+args.steps,
                steps_scope='GLNO yaw calibration only; ancestral stationary calibration excluded',
                run_steps=args.steps,
                dynamics_context=dict(recurrence=trial.recurrence,
                                      synapse_scale=network.parameters['synapse_scale'],
                                      cue_strength=network.parameters['cue_strength']),
                parameters=trial.parameters_dict(),state_dict=trial.state_dict(),
                optimizer=optimizer.state_dict(),learn_tau=learn_tau,
                speed_weight=args.speed_weight,run_loss_history=history,
                anatomy_sha256=digest(args.anatomy),phases_sha256=digest(args.phases),
                root_ids=network.root_ids,parent_checkpoint_sha256=digest(args.checkpoint),
                resumed_optimizer=args.resume,vehicle_integration_allowed=False,
                limitations=['Supervised ideal angular references only during optimization.',
                             'Sensory transduction, type signs and kinetics are hypotheses.',
                             'No claim of continuous attractor or validated physiology.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as stream:torch.save(result,stream)


if __name__=='__main__':
    main()
