"""Offline feasibility fitting of a continuous family of heading equilibria.

The auxiliary periodic curve is NOT a runtime state generator or new wiring.
Only physiological parameters are exported for replay. Individual threshold
offsets are an explicit optional hypothesis, not measured cell physiology.
Small equilibrium/tangent residuals do not establish transverse stability;
independent cue-and-release, perturbation and yaw trials are still required.
"""
import math
import torch
from torch import nn

from .heading_yaw import GLNOYawFit


class HeadingManifoldFit(GLNOYawFit):
    def __init__(self,network,*,harmonics=2,learn_neuron_bias=False):
        super().__init__(network)
        if learn_neuron_bias:
            offsets=(self.neuron_offsets.detach().clone() if hasattr(self,'neuron_offsets')
                     else torch.zeros(len(self.counts)))
            self.neuron_offsets=nn.Parameter(offsets)
        if isinstance(harmonics,bool) or not isinstance(harmonics,int) or not 1<=harmonics<=4:
            raise ValueError('one to four auxiliary Fourier harmonics required')
        self.harmonics=harmonics
        coefficients=torch.zeros(1+2*harmonics,len(self.counts))
        coefficients[0]=self._bias_values().detach()
        coefficients[1,self.cue_indices]=2*self.cue_phases.cos()
        coefficients[2,self.cue_indices]=2*self.cue_phases.sin()
        self.curve_coefficients=nn.Parameter(coefficients)

    def _basis(self,phases):
        if phases.ndim!=1 or not len(phases) or not torch.isfinite(phases).all():
            raise ValueError('finite nonempty phase batch required')
        values=[torch.ones_like(phases)];derivatives=[torch.zeros_like(phases)]
        for harmonic in range(1,self.harmonics+1):
            angle=harmonic*phases
            values.extend((angle.cos(),angle.sin()))
            derivatives.extend((-harmonic*angle.sin(),harmonic*angle.cos()))
        return torch.stack(values,dim=1),torch.stack(derivatives,dim=1)

    def curve(self,phases):
        basis,derivative=self._basis(phases)
        state=torch.sigmoid(basis@self.curve_coefficients)
        tangent=state*(1-state)*(derivative@self.curve_coefficients)
        return state,tangent

    def initialize_curve(self,phases,states):
        basis,_=self._basis(phases)
        if (len(phases)<1+2*self.harmonics or states.shape!=(len(phases),len(self.counts))
                or not torch.isfinite(states).all() or torch.any((states<0)|(states>1))
                or int(torch.linalg.matrix_rank(basis))!=basis.shape[1]):
            raise ValueError('full-rank sampled phases and bounded neural states required')
        with torch.no_grad():
            logits=torch.logit(states.clamp(1e-4,1-1e-4))
            self.curve_coefficients.copy_(torch.linalg.lstsq(basis,logits).solution)

    def curve_terms(self,phases):
        state,tangent=self.curve(phases)
        weights=self.weights()
        feedback=state@weights.T if self.recurrence else 0.
        target=torch.sigmoid(feedback+self._bias_values())
        equilibrium=(target-state).square().mean()
        response=target*(1-target)*(tangent@weights.T) if self.recurrence else torch.zeros_like(tangent)
        neutral=(response-tangent).square().mean()/tangent.square().mean().clamp_min(1e-6)
        rates=state@self.groups
        moment=(rates-rates.mean(1,keepdim=True))@self.moments
        magnitude=torch.linalg.vector_norm(moment,dim=1)
        rate_derivative=(target-state)/self.tau
        column_derivative=rate_derivative@self.groups
        moment_derivative=(column_derivative-column_derivative.mean(1,keepdim=True))@self.moments
        angular_drift=(moment[:,0]*moment_derivative[:,1]-moment[:,1]*moment_derivative[:,0]
                       )/magnitude.square().clamp_min(1e-12)
        desired=torch.stack((phases.cos(),phases.sin()),dim=1)
        phase_error=((moment/magnitude[:,None].clamp_min(1e-6)-desired)**2).mean()
        contrast=magnitude/rates.mean(1).clamp_min(1e-6)
        noncollapse=(.15-contrast).clamp_min(0).square().mean()
        return dict(equilibrium=equilibrium,neutral=neutral,phase=phase_error,
                    noncollapse=noncollapse,drift=angular_drift.square().mean())

    def curve_loss(self,phases):
        terms=self.curve_terms(phases)
        regularizer=.001*((self.log_gain-self.initial_log_gain).square().mean()
                          +(self.bias-self.initial_bias).square().mean())
        return (500*terms['equilibrium']+terms['neutral']+5*terms['phase']
                +100*terms['noncollapse']+regularizer)

    def release_loss(self,phases,*,cue_steps=100,dark_steps=100,phase_weight=0.):
        """Require actual cue-and-release dynamics to access the fitted curve.

        The curve is a training comparison only, not assigned to neural state.
        """
        target,_=self.curve(phases)
        if (any(isinstance(n,bool) or not isinstance(n,int) or n<1
                for n in (cue_steps,dark_steps))
                or not math.isfinite(phase_weight) or not 0<=phase_weight<=100):
            raise ValueError('positive integer simulation lengths and bounded phase weight required')
        state=torch.zeros_like(target);weights=self.weights();cue=self._cue(phases)
        for _ in range(cue_steps):state=self._advance(state,.01,weights,cue)
        phase_errors=[]
        desired=torch.stack((phases.cos(),phases.sin()),dim=1)
        for i in range(dark_steps):
            state=self._advance(state,.01,weights,0.)
            if phase_weight and ((i+1)%10==0 or i==dark_steps-1):
                rates=state@self.groups
                moment=(rates-rates.mean(1,keepdim=True))@self.moments
                unit=moment/torch.linalg.vector_norm(moment,dim=1,keepdim=True).clamp_min(1e-6)
                phase_errors.append((unit-desired).square().mean())
        phase_penalty=torch.stack(phase_errors).mean() if phase_errors else 0.
        return (state-target).square().mean()+phase_weight*phase_penalty
