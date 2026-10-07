"""Reproducible isolated heading tests; never imports car or best checkpoint."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from .heading_ring import HeadingRing
from .heading_connectome import _sha256


def _error(a,b):
    if a is None or b is None:
        return 180.
    return math.degrees(abs(math.atan2(math.sin(a-b),math.cos(a-b))))


def _advance(ring,seconds,velocity=0.,cue=None):
    for _ in range(round(seconds/.01)):
        ring.step(velocity,.01,cue_phase=cue)


def run_bench(*,phases=None,velocities=(-2.,-1.,-.5,-.2,-.05,.05,.2,.5,1.,2.)):
    phases=tuple(phases if phases is not None else
                 (-math.pi+(j+.37)*2*math.pi/12 for j in range(12)))
    holds,turns,reanchors=[],[],[]
    for phase in phases:
        ring=HeadingRing();_advance(ring,1.,cue=phase)
        initial=ring.heading;_advance(ring,10.)
        holds.append(dict(phase=phase,error_degrees=_error(ring.heading,initial),contrast=ring.contrast))
        for velocity in velocities:
            ring=HeadingRing();_advance(ring,1.,cue=phase)
            initial=ring.heading;_advance(ring,2.,velocity)
            turns.append(dict(phase=phase,velocity=velocity,
                              error_degrees=_error(ring.heading,initial+2*velocity),
                              contrast=ring.contrast))
        _advance(ring,2.,cue=-.8)
        reanchors.append(_error(ring.heading,-.8))
    lesion=HeadingRing(recurrence=False);_advance(lesion,1.,cue=.37);_advance(lesion,10.)
    no_turn=HeadingRing(turn_input=False);_advance(no_turn,1.,cue=.37)
    no_turn_initial=no_turn.heading;_advance(no_turn,2.,.5)
    long_turn=HeadingRing();_advance(long_turn,1.,cue=.37)
    initial=long_turn.heading;_advance(long_turn,20.,2.)
    return dict(anatomical_reconstruction=False,model='synthetic compass/paired-shift rate motif',
                inputs='ideal cue phase and angular velocity; no camera, car or goal',
                dt=.01,internal_max_dt=.002,
                dark_hold=dict(seconds=10,max_error_degrees=max(r['error_degrees'] for r in holds),cases=holds),
                turns=turns,max_turn_error_degrees=max(r['error_degrees'] for r in turns),
                reanchor=dict(seconds=2,max_error_degrees=max(reanchors)),
                ablations=dict(without_recurrence_contrast=lesion.contrast,
                               without_turn_input_error_degrees=_error(no_turn.heading,no_turn_initial+1)),
                long_max_speed_turn=dict(seconds=20,velocity=2.,
                                         error_degrees=_error(long_turn.heading,initial+40),
                                         contrast=long_turn.contrast),
                limitations=['Synthetic sectors and effective signed feedback, not anatomical weights.',
                             'Neutral first-harmonic memory, not robust amplitude-restoring bump attraction.',
                             'Ideal cue adapter bypasses the visual system.',
                             'Only noiseless inputs tested; small-rate nonlinearities and long-run drift remain.',
                             'No proof of spontaneous bump formation or biological neuron fidelity.',
                             'Not connected to deployed motor control.'])


def annotation_audit(anatomy_path,annotations_path):
    anatomy=json.loads(Path(anatomy_path).read_text())
    ids={n['root_id'] for n in anatomy['neurons']}
    with open(annotations_path,newline='') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if not {'root_id','hemibrain_type','side','top_nt'} <= set(reader.fieldnames or ()):
            raise ValueError('missing annotation columns')
        fields=reader.fieldnames
        selected=[r for r in reader if r['root_id'] in ids]
    if len(selected)!=len(ids) or len({r['root_id'] for r in selected})!=len(ids):
        raise ValueError('missing or duplicate selected annotation IDs')
    return dict(source='https://github.com/flyconnectome/flywire_annotations/tree/v2.1.0',
                annotations_sha256=_sha256(annotations_path),anatomy_sha256=_sha256(anatomy_path),
                selected_neurons=len(ids),matched=len(selected),available_fields=fields,
                type_counts=dict(Counter(r['hemibrain_type'] for r in selected)),
                sides=dict(Counter(r['side'] for r in selected)),
                predicted_transmitters=dict(Counter(r['top_nt'] for r in selected)),
                sector_assignment_available=False,
                notes=['Representative and soma XYZ positions are not EB/PB terminal sectors.',
                       'Predicted transmitter labels are not validated postsynaptic signs.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--anatomy',type=Path)
    parser.add_argument('--annotations',type=Path)
    args=parser.parse_args()
    if bool(args.anatomy)!=bool(args.annotations):
        parser.error('--anatomy and --annotations must be supplied together')
    report=run_bench()
    if args.anatomy:
        report['annotation_audit']=annotation_audit(args.anatomy,args.annotations)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('max_turn_error_degrees','ablations','long_max_speed_turn')}))


if __name__=='__main__':
    main()
