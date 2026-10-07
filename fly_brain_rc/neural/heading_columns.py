"""Offline morphological PB-column hypotheses; never anatomical ground truth."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

NBLAST_MD5 = '3ca6d79615e1cd30b37f32a8ae845f32'
META_SHA256 = 'cf1c6d1821cf9179584e2ba89a614bd1b184234e547b660d378c964ed46fd9e1'
NBLAST_URL = 'https://zenodo.org/records/10877326'
META_URL = ('https://raw.githubusercontent.com/flyconnectome/flywire_annotations/'
            'v2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv')
ALIASES = {'PEN_a/PEN1': 'PEN_a(PEN1)', 'PEN_b/PEN2': 'PEN_b(PEN2)'}


def selected_neurons(anatomy):
    selected = {}
    for neuron in anatomy['neurons']:
        root = neuron['root_id']
        if (not isinstance(root, str) or not root.isdigit() or root in selected
                or neuron['cell_type'] not in ('EPG', 'PEG', 'Delta7', *ALIASES)):
            raise ValueError('invalid selected neuron identifier or cell type')
        selected[root] = neuron
    if not selected:
        raise ValueError('empty selected circuit')
    return selected


def compatible_matches(kind, bodies, values, meta):
    if len(bodies) != len(values):
        raise ValueError('misaligned NBLAST identifiers and scores')
    kind = ALIASES.get(kind, kind)
    return [(body, meta[body]['instance'], score)
            for body, score in zip(bodies, values)
            if body in meta and meta[body]['type'] == kind]


def columns(instance):
    """Parse innervation suffix, not PBxx type code or soma-side suffix."""
    suffix = instance.partition(')_')[2].split('_')[0]
    if not re.fullmatch(r'(?:[LR][1-9])+', suffix):
        return ()
    return tuple(re.findall(r'[LR][1-9]', suffix))


def rank_columns(rows, minimum_score=.3, minimum_margin=.05):
    """Group compatible matches by column set before computing ambiguity.

    Thresholds are engineering filters, NOT calibrated probabilities.
    No one-to-one cross-specimen correspondence is assumed.
    """
    if not (0 <= minimum_score <= 1 and 0 < minimum_margin <= 1):
        raise ValueError('invalid hypothesis thresholds')
    groups = {}
    for body, instance, score in rows:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid NBLAST similarity')
        key = columns(instance)
        if not key:
            continue
        candidate = dict(columns=list(key), score=float(score),
                         hemibrain_body_id=str(body), instance=instance)
        if key not in groups or score > groups[key]['score']:
            groups[key] = candidate
    ranked = sorted(groups.values(), key=lambda row: -row['score'])
    if not ranked:
        return dict(status='unavailable', candidate_columns=None,
                    margin=None, alternatives=[])
    # Absence of another column is not evidence of a large margin.
    margin = ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None
    supported = (ranked[0]['score'] >= minimum_score and
                 margin is not None and margin >= minimum_margin)
    return dict(status=('morphology_supported_hypothesis' if supported else 'ambiguous'),
                candidate_columns=ranked[0]['columns'] if supported else None,
                margin=margin, alternatives=ranked)


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def map_columns(anatomy_path, meta_path, nblast_path,
                minimum_score=.3, minimum_margin=.05):
    if digest(nblast_path, 'md5') != NBLAST_MD5:
        raise ValueError('NBLAST checksum differs from published v783 dataset')
    if digest(meta_path) != META_SHA256:
        raise ValueError('hemibrain metadata checksum differs from pinned v2.1.0')
    # Optional dependency is imported only for this offline tool.
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc

    anatomy = json.loads(Path(anatomy_path).read_text())
    with open(meta_path, newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'bodyId', 'instance', 'type'} <= set(reader.fieldnames or ()):
            raise ValueError('invalid hemibrain metadata columns')
        meta = {row['bodyId']: row for row in reader}
    schema = ipc.open_file(str(nblast_path)).schema
    selected = selected_neurons(anatomy)
    field_names = {}
    for name in schema.names:
        root, separator, sv = name.partition(',')
        if separator and root in selected:
            if root in field_names:
                raise ValueError('duplicate selected FlyWire root in NBLAST')
            field_names[root] = name
    if 'query' not in schema.names:
        raise ValueError('missing hemibrain query identifier column')
    table = feather.read_table(str(nblast_path), columns=['query', *field_names.values()])
    body_ids = [str(value) for value in table['query'].to_pylist()]
    compatible = {ALIASES.get(n['cell_type'], n['cell_type']) for n in selected.values()}
    indices = [i for i, body in enumerate(body_ids)
               if body in meta and meta[body]['type'] in compatible]
    table = table.take(indices)
    bodies = [str(value) for value in table['query'].to_pylist()]
    results = []
    for root, neuron in selected.items():
        matches = []
        if root in field_names:
            values = table[field_names[root]].to_pylist()
            matches = compatible_matches(neuron['cell_type'], bodies, values, meta)
        results.append(dict(root_id=root, cell_type=neuron['cell_type'],
                            **rank_columns(matches, minimum_score, minimum_margin)))
    return dict(version=1, functional_model=False, validated_assignments=0,
                neurons=results, status_counts=dict(Counter(r['status'] for r in results)),
                thresholds=dict(minimum_score=minimum_score, minimum_margin=minimum_margin),
                sources=dict(nblast_url=NBLAST_URL, nblast_md5=NBLAST_MD5,
                             metadata_url=META_URL, metadata_sha256=digest(meta_path),
                             anatomy_sha256=digest(anatomy_path)),
                limitations=['Cross-specimen morphology hypotheses, not verified identities.',
                             'NBLAST scores and margins are not confidence probabilities.',
                             'Unmirrored matches only; missing matches remain unavailable.',
                             'PB columns are not automatically EB angles or input phases.',
                             'No dynamics or deployed vehicle changes.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('anatomy', 'metadata', 'nblast', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--minimum-score', type=float, default=.3)
    parser.add_argument('--minimum-margin', type=float, default=.05)
    args = parser.parse_args()
    result = map_columns(args.anatomy, args.metadata, args.nblast,
                         args.minimum_score, args.minimum_margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['status_counts']))


if __name__ == '__main__':
    main()
