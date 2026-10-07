"""Bounded stationary-memory pilot; no claimed yaw adapter or vehicle promotion."""
import argparse
import json
import math
from pathlib import Path

from .anatomical_heading import AnatomicalHeading, pilot_parameters
from .heading_columns import digest

DEFAULT_DIRECTIONS=tuple(-math.pi+(i+.37)*2*math.pi/12 for i in range(12))


def _error(a,b):
    return 180. if a is None else math.degrees(abs(math.atan2(math.sin(a-b),math.cos(a-b))))


def evaluate(anatomy,phases,parameters,*,cue_seconds=1.,dark_seconds=3.,
             directions=DEFAULT_DIRECTIONS,recurrence=True):
    if not all(math.isfinite(v) and v>=.01 for v in (cue_seconds,dark_seconds)):
        raise ValueError('cue and dark intervals must be finite and at least .01 seconds')
    directions=tuple(directions)
    if not directions or not all(math.isfinite(v) for v in directions):
        raise ValueError('finite directions required')
    # Require exact multiples of the fixed outer time step; never misreport duration.
    for seconds in (cue_seconds,dark_seconds):
        if abs(seconds/.01-round(seconds/.01))>1e-8:
            raise ValueError('intervals must be multiples of .01 seconds')
    cases=[]
    for phase in directions:
        model=AnatomicalHeading(anatomy,phases,parameters,recurrence=recurrence)
        for _ in range(round(cue_seconds/.01)):
            model.step(.01,cue_phase=phase)
        initial=model.heading
        cue_error=_error(initial,phase)
        maximum_dark_error=0.;minimum_dark_contrast=float('inf')
        for _ in range(round(dark_seconds/.01)):
            model.step(.01)
            maximum_dark_error=max(maximum_dark_error,_error(model.heading,phase))
            minimum_dark_contrast=min(minimum_dark_contrast,model.contrast)
        rates=model.activity
        cases.append(dict(cue_phase=phase,cue_heading=initial,
                          cue_error_degrees=cue_error,dark_heading=model.heading,
                          dark_error_degrees=_error(model.heading,phase),
                          dark_contrast=model.contrast,
                          maximum_dark_error_degrees=maximum_dark_error,
                          minimum_dark_contrast=minimum_dark_contrast,
                          maximum_rate=float(rates.max()),minimum_rate=float(rates.min())))
    passed=all(c['cue_error_degrees']<30 and c['maximum_dark_error_degrees']<30
               and c['minimum_dark_contrast']>.1 for c in cases)
    return dict(model='fixed directed count matrix, signed type-pair gains, sigmoid rates',
                parameters=model.parameters,neurons=len(model.root_ids),
                synapses=int(model.synapse_counts.sum()),regional_rows=len(anatomy['connections']),
                mapped_epg_neurons=sum(n.get('phase_degrees') is not None for n in phases['neurons']),
                cue_seconds=cue_seconds,dark_seconds=dark_seconds,dt=.01,
                recurrence=recurrence,cases=cases,memory_gate_passed=passed,
                memory_gate=dict(maximum_cue_error_degrees=30,maximum_dark_error_degrees=30,
                                 minimum_dark_contrast=.1),
                diagnostic_score=sum(c['cue_error_degrees']+c['maximum_dark_error_degrees']
                                     for c in cases)/len(cases),
                yaw_integration_implemented=False,
                validated_physiology=False)


def sweep(anatomy,phases,*,excitations=(2.,4.,8.,12.),
          delta7_gains=(-2.,-4.,-8.,-12.,4.),**kwargs):
    candidates=[evaluate(anatomy,phases,pilot_parameters(excitation=exc,delta7_gain=delta),
                         **kwargs) for exc in excitations for delta in delta7_gains]
    if not candidates:
        raise ValueError('empty parameter grid')
    selected=min(range(len(candidates)),key=lambda i:candidates[i]['diagnostic_score'])
    return dict(candidates=candidates,selected_diagnostic_index=selected,
                passed_candidates=sum(c['memory_gate_passed'] for c in candidates),
                vehicle_integration_allowed=False,
                selection='minimum mean cue error + maximum dark error; diagnostic only',
                limitations=['Incomplete induced heading circuit; GLNO and ring populations absent.',
                             'Signed type-pair gains and kinetics are explicit engineering hypotheses.',
                             'Only shared excitation/Delta7 gains swept; no exhaustive type-pair fitting.',
                             'EPG phases remain cross-specimen morphology hypotheses.',
                             'Ideal cue directly modulates EPG, bypassing visual input circuitry.',
                             'No yaw input, no vehicle control, no neural recording validation.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anatomy',type=Path,required=True)
    parser.add_argument('--phases',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    anatomy=json.loads(args.anatomy.read_text());phases=json.loads(args.phases.read_text())
    result=sweep(anatomy,phases)
    candidate=result['candidates'][result['selected_diagnostic_index']]
    # New directions and a full sixty-second dark challenge.
    held_out=tuple(-math.pi+(i+.87)*2*math.pi/12 for i in range(12))
    result['long_hold']=evaluate(anatomy,phases,candidate['parameters'],
                                 dark_seconds=60.,directions=held_out)
    result['recurrence_lesion']=evaluate(anatomy,phases,candidate['parameters'],recurrence=False)
    result['sources']=dict(anatomy_sha256=digest(args.anatomy),phases_sha256=digest(args.phases))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(candidates=len(result['candidates']),passed=result['passed_candidates'],
                          selected=result['selected_diagnostic_index'],
                          score=candidate['diagnostic_score'],
                          long_hold_passed=result['long_hold']['memory_gate_passed'])))


if __name__=='__main__':
    main()
