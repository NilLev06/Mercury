"""Synthetic compass/shift population motif for isolated functional testing.

Inspired by CX ring-attractor mechanisms, NOT a FlyWire reconstruction or
an exact reproduction of a published model. Signed effective feedback stands
in for excitation and inhibition. Sector assignments are synthetic; no root
IDs are assigned to these units. No scalar heading is integrated or stored.
At rest the first harmonic has neutral gain, not an amplitude-restoring
biological bump attractor. Turning can change amplitude and accumulate drift.
"""
import math
import numpy as np


class HeadingRing:
    SECTORS=16
    MAX_YAW_RATE=2.0
    TAU_COMPASS=.05
    TAU_SHIFT=.02

    def __init__(self,*,recurrence=True,turn_input=True):
        self.recurrence=recurrence
        self.turn_input=turn_input
        self.preferred=np.linspace(-math.pi,math.pi,self.SECTORS,endpoint=False)
        delta=2*math.pi/self.SECTORS
        difference=self.preferred[:,None]-self.preferred[None,:]
        # Paired offset loops preserve the first harmonic at rest. Positive
        # yaw increases the loop projecting in the positive phase direction.
        self._positive=np.cos(difference-delta)/(self.SECTORS*math.cos(delta))
        self._negative=np.cos(difference+delta)/(self.SECTORS*math.cos(delta))
        self._turn_gain=(self.TAU_COMPASS+self.TAU_SHIFT)/math.tan(delta)
        self._compass=np.ones(self.SECTORS)
        self._left=np.ones(self.SECTORS)
        self._right=np.ones(self.SECTORS)

    @property
    def activity(self):
        return self._compass.copy()

    @property
    def contrast(self):
        vector=np.dot(self._compass,np.exp(1j*self.preferred))
        return float(abs(vector)/max(float(self._compass.sum()),1e-12))

    @property
    def heading(self):
        if self.contrast<1e-4:
            return None
        vector=np.dot(self._compass,np.exp(1j*self.preferred))
        return float(np.angle(vector))

    def step(self,yaw_rate,dt,*,cue_phase=None):
        if (not math.isfinite(yaw_rate) or abs(yaw_rate)>self.MAX_YAW_RATE
                or not math.isfinite(dt) or not 0<dt<=1.
                or (cue_phase is not None and not math.isfinite(cue_phase))):
            raise ValueError('finite supported yaw rate, cue and 0<dt<=1 required')
        velocity=yaw_rate if self.turn_input else 0.
        gate=self._turn_gain*velocity
        cue=None if cue_phase is None else 1+.8*np.cos(self.preferred-cue_phase)
        substeps=max(1,math.ceil(dt/.002))
        h=dt/substeps
        for _ in range(substeps):
            feedback=(self._positive@self._left+self._negative@self._right
                      if self.recurrence else np.zeros(self.SECTORS))
            drive=1+feedback
            # Ideal sensory anchoring adapter, not a model of ER neurons or
            # a camera decoder. Visibility gates attraction toward its profile.
            if cue is not None:
                drive+=2*(cue-self._compass)
            e=self._compass+h/self.TAU_COMPASS*(-self._compass+np.maximum(0,drive))
            left=self._left+h/self.TAU_SHIFT*(-self._left+(1+gate)*self._compass)
            right=self._right+h/self.TAU_SHIFT*(-self._right+(1-gate)*self._compass)
            self._compass,self._left,self._right=e,left,right
