"""Trajectory-wide offline yaw checks. No promotion or vehicle mutation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .anatomical_heading import AnatomicalHeading,extend_phases
from .heading_columns import digest
from .heading_yaw import GLNOYawFit


def sensitivity_scores(lower,upper,contrast,offset):
    """Small initial differences must survive, not coalesce into fixed headings."""
    lower=np.asarray(lower);upper=np.asarray(upper);contrast=np.asarray(contrast)
    if (lower.ndim!=1 or not len(lower) or upper.shape!=lower.shape
            or contrast.shape!=(2,len(lower))
            or not all(np.isfinite(x).all() for x in (lower,upper,contrast))
            or np.any(contrast<0) or not math.isfinite(offset) or not 0<offset<=.1):
        raise ValueError('finite paired angles, contrasts and offset in (0,.1] required')
    difference=np.arctan2(np.sin(upper-lower),np.cos(upper-lower))
    sensitivity=difference/(2*offset)
    valid=(contrast>=.05).all(axis=0)
    good=valid&(sensitivity>=.5)&(sensitivity<=1.5)
    return dict(median_sensitivity=float(np.median(sensitivity)),
                collapsed_pairs=int((np.abs(sensitivity)<.05).sum()),
                invalid_pairs=int((~valid).sum()),pairs=len(lower),
                fraction_near_unit_sensitivity=float(good.mean()),
                passed=bool(valid.all() and good.mean()>=.95))


def evaluate_sensitivity(anatomy,phases,parameters,*,seconds=60.,offset=.01,directions=None):
    directions=tuple(directions if directions is not None else
                     (-math.pi+.217+i*math.pi/36 for i in range(72)))
    if (not directions or not all(math.isfinite(v) for v in directions)
            or not math.isfinite(seconds) or not .01<=seconds<=60
            or abs(seconds/.01-round(seconds/.01))>1e-8
            or not math.isfinite(offset) or not 0<offset<=.1):
        raise ValueError('finite directions, bounded duration and small positive offset required')
    trial=GLNOYawFit(AnatomicalHeading(anatomy,phases,parameters))
    phase=torch.tensor(tuple(v-offset for v in directions)+tuple(v+offset for v in directions))
    with torch.no_grad():
        state=torch.zeros(len(phase),len(trial.counts))
        for _ in range(100):state=trial.step(state,.01,cue_phase=phase)
        for _ in range(round(seconds/.01)):state=trial.step(state,.01)
        rates=state@trial.groups
        moment=(rates-rates.mean(1,keepdim=True))@trial.moments
        angle=torch.atan2(moment[:,1],moment[:,0]).numpy().reshape(2,-1)
        contrast=(torch.linalg.vector_norm(moment,dim=1)/rates.mean(1).clamp_min(1e-12)).numpy().reshape(2,-1)
    return dict(**sensitivity_scores(angle[0],angle[1],contrast,offset),
                seconds=seconds,offset_radians=offset,vehicle_integration_allowed=False)


def trajectory_scores(actual,reference,contrast,dt):
    """Score wrapped decoded angles against an unwrapped reference trajectory."""
    actual=np.asarray(actual);reference=np.asarray(reference);contrast=np.asarray(contrast)
    if (actual.ndim!=2 or actual.shape[0]<2 or actual.shape[1]<1
            or reference.shape!=actual.shape or contrast.shape!=actual.shape
            or not all(np.isfinite(v).all() for v in (actual,reference,contrast))
            or np.any(contrast<0) or not math.isfinite(dt) or not 0<dt<=1):
        raise ValueError('matching finite time x trial traces and valid dt required')
    wrap=lambda angle:np.arctan2(np.sin(angle),np.cos(angle))
    error=np.degrees(np.abs(wrap(actual-reference)))
    invalid=contrast<.05
    error[invalid]=180.
    increments=wrap(np.diff(actual,axis=0))
    expected=np.diff(reference,axis=0)
    rmse=float(np.sqrt(np.mean(((increments-expected)/dt)**2)))
    maximum=float(error.max())
    return dict(mean_final_error_degrees=float(error[-1].mean()),maximum_error_degrees=maximum,
                minimum_contrast=float(contrast.min()),invalid_samples=int(invalid.sum()),
                speed_rmse_rad_s=rmse,maximum_step_degrees=float(np.degrees(np.abs(increments)).max()),
                passed=bool(maximum<10. and rmse<.1 and not invalid.any()))


def evaluate_yaw(anatomy,phases,parameters,*,directions=None,segments=((4.,.4),(2.,0.))):
    directions=tuple(directions if directions is not None else
                     (-math.pi+.43+i*math.pi/6 for i in range(12)))
    segments=tuple(segments)
    if not directions or not all(math.isfinite(v) for v in directions) or not segments:
        raise ValueError('finite directions and a nonempty schedule required')
    if any(not math.isfinite(seconds) or not .01<=seconds<=60
           or abs(seconds/.01-round(seconds/.01))>1e-8
           or not math.isfinite(yaw) or abs(yaw)>5 for seconds,yaw in segments):
        raise ValueError('bounded schedules at .01s resolution required')
    trial=GLNOYawFit(AnatomicalHeading(anatomy,phases,parameters))
    phase=torch.tensor(directions,dtype=torch.float32)
    actual=[];references=[];contrasts=[]
    with torch.no_grad():
        state=torch.zeros(len(phase),len(trial.counts))
        for _ in range(100):state=trial.step(state,.01,cue_phase=phase)
        reference=phase.clone()
        def record():
            rates=state@trial.groups
            moment=(rates-rates.mean(dim=1,keepdim=True))@trial.moments
            actual.append(torch.atan2(moment[:,1],moment[:,0]).numpy())
            contrasts.append((torch.linalg.vector_norm(moment,dim=1)/rates.mean(
                dim=1).clamp_min(1e-12)).numpy())
            references.append(reference.numpy().copy())
        record()
        for seconds,yaw in segments:
            velocity=torch.full_like(phase,yaw)
            for _ in range(round(seconds/.01)):
                state=trial.step_yaw(state,.01,velocity)
                # Ground truth is used for scoring only, never neural input.
                reference=reference+yaw*.01
                record()
    result=trajectory_scores(actual,references,contrasts,.01)
    result.update(segments=segments,directions=len(directions),dt=.01,cue_seconds=1.,
                  vehicle_integration_allowed=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('anatomy','phases','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,weights_only=True)
    if (checkpoint.get('anatomy_sha256')!=digest(args.anatomy)
            or checkpoint.get('phases_sha256')!=digest(args.phases)):
        raise ValueError('checkpoint input provenance does not match')
    anatomy=json.loads(args.anatomy.read_text())
    phases=extend_phases(anatomy,json.loads(args.phases.read_text()))
    torch.set_num_threads(1)
    cases=dict(stationary=((60.,0.),),slow=((10.,.05),(2.,0.)),
               forward=((4.,.4),(2.,0.)),reverse=((4.,-.4),(2.,0.)),
               reversal=((2.,.4),(2.,-.4),(2.,0.)))
    for name,segments in cases.items():
        result=evaluate_yaw(anatomy,phases,checkpoint['parameters'],segments=segments)
        print(json.dumps(dict(case=name,**result)),flush=True)
    print(json.dumps(dict(case='pinning',**evaluate_sensitivity(
        anatomy,phases,checkpoint['parameters']))),flush=True)


if __name__=='__main__':
    main()
