"""Published EPG PB-to-EB phases applied to uncertain morphology hypotheses.

Not dynamics, not independently verified FlyWire terminal assignments, and not
a shared angular map for PEN/PEG/Delta7. Arbitrary global phase/sign convention
must be aligned with sensory and yaw conventions before vehicle integration.
"""
import argparse
import json
from pathlib import Path

from .heading_columns import digest

SOURCE = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10881393/#Fig11'
FIGURE = 'Extended Data Figure 5c, EPG mapping'
# Mussells Pires et al., Nature 2024, DOI 10.1038/s41586-023-07006-3.
# Transcribed from the original image, not root ordering or soma coordinates.
# L9/R9 include out-of-scope EPGt and are deliberately absent here.
EPG_PB_PHASES = {
    'L1': -11.25, 'L2': -56.25, 'L3': -101.25, 'L4': -146.25,
    'L5': 168.75, 'L6': 123.75, 'L7': 78.75, 'L8': 33.75,
    'R1': 11.25, 'R2': -33.75, 'R3': -78.75, 'R4': -123.75,
    'R5': -168.75, 'R6': 146.25, 'R7': 101.25, 'R8': 56.25,
}


def epg_phase(column):
    return EPG_PB_PHASES.get(column)


def phase_hypotheses(mapping):
    neurons = []
    for neuron in mapping['neurons']:
        cols = neuron.get('candidate_columns')
        phase = None
        if (neuron['cell_type'] == 'EPG' and
                neuron['status'] == 'morphology_supported_hypothesis' and
                cols is not None and len(cols) == 1):
            phase = epg_phase(cols[0])
        neurons.append(dict(root_id=neuron['root_id'], cell_type=neuron['cell_type'],
                            phase_degrees=phase,
                            status='phase_hypothesis' if phase is not None else 'unresolved',
                            candidate_columns=cols))
    return dict(version=1, functional_model=False, validated_assignments=0,
                neurons=neurons, epg_phases=sum(n['phase_degrees'] is not None for n in neurons),
                source=dict(url=SOURCE, figure=FIGURE, pb_phase_table=EPG_PB_PHASES),
                limitations=['Cross-specimen PB assignment uncertainty is retained.',
                             'Only selected EPG phases, not PEN, PEG or Delta7 phase assignments.',
                             'Angle zero and sign are source conventions, not physical vehicle orientation.',
                             'No independently verified FlyWire terminal morphology or anatomical dynamics.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = phase_hypotheses(json.loads(args.mapping.read_text()))
    result['mapping_sha256'] = digest(args.mapping)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(epg_phase_hypotheses=result['epg_phases'], validated_assignments=0)))


if __name__ == '__main__':
    main()
