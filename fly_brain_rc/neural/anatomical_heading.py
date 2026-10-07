"""Isolated rate dynamics on measured directed FlyWire heading connections.

Topology/counts are anatomical; gains, biases, time constants, sign hypotheses
and ideal cue are not validated physiology. No camera, yaw adapter or car control.
"""
import copy
import math
import numpy as np

from .heading_connectome import HEADING_TYPES, HEADING_INPUT_TYPES
from .heading_columns import selected_neurons


def pilot_parameters(excitation=4., delta7_gain=-4.):
    """Unfitted engineering starting point; negative Delta7 is an assumption."""
    return dict(gain={post:{pre:float(delta7_gain if pre=='Delta7' else excitation)
                           for pre in HEADING_TYPES} for post in HEADING_TYPES},
                bias={kind:-2. for kind in HEADING_TYPES},
                tau={kind:.05 for kind in HEADING_TYPES},
                synapse_scale=1000., cue_strength=6.)


def expanded_parameters(anatomy):
    """Explicit sign hypotheses, not receptor inference from predicted NT.

    ER and GLNO: inhibitory; cholinergic ExR4/7/8: excitatory. Mixed ExR1/6
    treated as inhibitory. Dopaminergic ExR2/3 treated as excitatory effective
    inputs for this pilot only; real neuromodulation is NOT represented.
    """
    kinds=tuple(sorted({n['cell_type'] for n in anatomy['neurons']}))
    negative={'Delta7','GLNO','ExR1','ExR5','ExR6'}
    negative.update(k for k in kinds if k.startswith('ER'))
    return dict(gain={post:{pre:-4. if pre in negative else 4. for pre in kinds}
                      for post in kinds},
                bias={k:-2. for k in kinds},tau={k:.05 for k in kinds},
                synapse_scale=1000.,cue_strength=6.)


def extend_phases(anatomy,phases):
    """Keep existing EPG evidence; new inputs receive no fabricated angles."""
    old={row['root_id']:row for row in phases['neurons']}
    if len(old)!=len(phases['neurons']):
        raise ValueError('duplicate phase identities')
    neurons=selected_neurons(anatomy)
    if not set(old)<=set(neurons):
        raise ValueError('phase identity missing from expanded graph')
    result=copy.deepcopy(phases)
    result['neurons']=[]
    for root,neuron in neurons.items():
        if root in old:
            row=copy.deepcopy(old[root])
            if row['cell_type']!=neuron['cell_type']:
                raise ValueError('phase type changed in expanded graph')
        elif neuron['cell_type'] in HEADING_INPUT_TYPES:
            row=dict(root_id=root,cell_type=neuron['cell_type'],phase_degrees=None,
                     status='unresolved',candidate_columns=None)
        else:
            raise ValueError('core neuron has no prior phase audit')
        result['neurons'].append(row)
    return result


class AnatomicalHeading:
    DECODE_THRESHOLD=.05

    def __init__(self, anatomy, phases, parameters, *, recurrence=True):
        neurons=selected_neurons(anatomy)
        self.root_ids=tuple(neurons)
        self._index={root:i for i,root in enumerate(self.root_ids)}
        self._types=tuple(neurons[root]['cell_type'] for root in self.root_ids)
        classes=parameters.get('cell_classes')
        if classes is not None:
            if set(classes)!=set(self.root_ids) or any(
                    classes[root] not in (f'{kind}:left',f'{kind}:right')
                    for root,kind in zip(self.root_ids,self._types)):
                raise ValueError('cell classes must match neuron identities and anatomical types')
        self._parameter_types=tuple(classes[root] for root in self.root_ids) if classes else self._types
        counts=np.zeros((len(neurons),len(neurons)),dtype=np.int64)
        for row in anatomy['connections']:
            pre,post,count=row['pre'],row['post'],row['syn_count']
            if (pre not in neurons or post not in neurons or isinstance(count,bool)
                    or not isinstance(count,int) or count<=0):
                raise ValueError('invalid anatomical edge or synapse count')
            counts[self._index[post],self._index[pre]]+=count
        if not counts.any():
            raise ValueError('empty anatomical connectivity')
        self._counts=counts
        self._parameters=copy.deepcopy(parameters)
        kinds=set(self._parameter_types)
        for field in ('gain','bias','tau'):
            if set(parameters[field])!=kinds:
                raise ValueError('parameters must cover exactly the selected cell types')
        for row in parameters['gain'].values():
            if set(row)!=kinds or not all(math.isfinite(v) and abs(v)<=100 for v in row.values()):
                raise ValueError('invalid type-pair gains')
        if not all(math.isfinite(v) and abs(v)<=100 for v in parameters['bias'].values()):
            raise ValueError('invalid type biases')
        if not all(math.isfinite(v) and .005<=v<=1 for v in parameters['tau'].values()):
            raise ValueError('supported time constants are 0.005 to 1 seconds')
        scale=parameters['synapse_scale'];strength=parameters['cue_strength']
        if not math.isfinite(scale) or scale<=0 or not math.isfinite(strength) or not 0<=strength<=100:
            raise ValueError('invalid scale or cue strength')
        gains=np.array([[parameters['gain'][post][pre] for pre in self._parameter_types]
                        for post in self._parameter_types])
        self._weights=counts/scale*gains
        self._weights.flags.writeable=False
        self._bias=np.array([parameters['bias'][kind] for kind in self._parameter_types])
        self._tau=np.array([parameters['tau'][kind] for kind in self._parameter_types])
        self._cue_strength=strength
        phase_by_id={}
        for row in phases['neurons']:
            root=row['root_id'];value=row.get('phase_degrees')
            if (root not in neurons or root in phase_by_id
                    or row['cell_type']!=neurons[root]['cell_type']):
                raise ValueError('phase identities do not match anatomical neurons')
            if value is not None and (row['cell_type']!='EPG'
                    or row.get('status')!='phase_hypothesis' or not math.isfinite(value)):
                raise ValueError('only finite EPG phase hypotheses are supported')
            phase_by_id[root]=value
        if set(phase_by_id)!=set(neurons):
            raise ValueError('missing anatomical neuron in phase report')
        groups={}
        self._cue_indices=[];self._cue_phases=[]
        for root,value in phase_by_id.items():
            if value is not None:
                angle=math.radians(round(value % 360.,8))
                groups.setdefault(angle,[]).append(self._index[root])
                self._cue_indices.append(self._index[root]);self._cue_phases.append(angle)
        if len(groups)<4:
            raise ValueError('at least four distinct EPG phase hypotheses required')
        self._groups=tuple(groups.values())
        self._directions=np.exp(1j*np.array(list(groups)))
        self._cue_indices=np.array(self._cue_indices)
        self._cue_phases=np.array(self._cue_phases)
        self._activity=np.zeros(len(neurons))
        self.recurrence=recurrence

    @property
    def parameters(self):
        return copy.deepcopy(self._parameters)

    @property
    def synapse_counts(self):
        return self._counts.copy()

    @property
    def activity(self):
        return self._activity.copy()

    def _decode(self):
        # One contribution per column, not per neuron; center out constant rates.
        rates=np.array([self._activity[indices].mean() for indices in self._groups])
        vector=np.dot(rates-rates.mean(),self._directions)
        contrast=float(abs(vector)/max(rates.sum(),1e-12))
        return float(np.angle(vector)) if contrast>=self.DECODE_THRESHOLD else None,contrast

    @property
    def heading(self):
        return self._decode()[0]

    @property
    def contrast(self):
        return self._decode()[1]

    def step(self, dt, *, cue_phase=None, external_drive=None):
        if (not math.isfinite(dt) or not 0<dt<=1
                or (cue_phase is not None and not math.isfinite(cue_phase))):
            raise ValueError('finite cue and 0<dt<=1 required')
        cue=np.zeros(len(self.root_ids))
        if external_drive is not None:
            drive=np.asarray(external_drive,dtype=float)
            if drive.shape!=cue.shape or not np.isfinite(drive).all() or np.any(np.abs(drive)>100):
                raise ValueError('finite neuron-sized external drive bounded by 100 required')
            cue+=drive
        if cue_phase is not None:
            # Ideal modulation around baseline, not an anatomical visual pathway.
            cue[self._cue_indices]+=self._cue_strength*np.cos(self._cue_phases-cue_phase)
        steps=max(1,math.ceil(dt/min(.01,float(self._tau.min())/5)))
        h=dt/steps
        for _ in range(steps):
            feedback=(np.einsum('ij,j->i',self._weights,self._activity)
                      if self.recurrence else 0.)
            target=1/(1+np.exp(-np.clip(feedback+self._bias+cue,-60,60)))
            self._activity+=h/self._tau*(target-self._activity)
