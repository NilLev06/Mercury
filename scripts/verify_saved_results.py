"""Verify the published paired results without running or training a simulator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/2026-10-14-classical-comparison'


def verify(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    rows, manifest, summary = read('final-rollouts.json'), read('final-manifest.json'), read('summary.json')
    selection, recorded = read('selection.json'), read('VERIFICATION.json')
    if len(rows) != 192:
        raise ValueError('Expected 192 reports: two controllers on 96 scenes')
    scenes = {case['name']: case for case in manifest['cases']}
    if len(scenes) != 96 or len(manifest['cases']) != 96:
        raise ValueError('Expected 96 unique scenes')
    pairs = Counter((r['scenario'], r['model']) for r in rows)
    if set(pairs) != {(name, model) for name in scenes for model in ('neural', 'classical')} or any(v != 1 for v in pairs.values()):
        raise ValueError('Each scene must have exactly one report per controller')
    for row in rows:
        spec = scenes[row['scenario']]
        if row['tier'] != spec['tier'] or row['map_seed'] != spec['map_seed']:
            raise ValueError('Report does not match its scene')
        if row['clean_success'] != bool(row['success'] and row['contacts'] == 0) or row['beta'] is not None:
            raise ValueError('Invalid clean-success criterion or teacher present')
        if not 0 < row['steps'] <= 2400 or abs(row['seconds'] - .02 * row['steps']) > 1e-8:
            raise ValueError('Invalid simulation duration')
        if row['success'] and not (row['final_goal_distance'] <= .5 and abs(row['final_speed']) <= .03 and row['stable_stop_seconds'] >= .5 - 1e-8):
            raise ValueError('Arrival does not satisfy the stopping criterion')
    for tier in (None, 'simple', 'complex'):
        for model in ('neural', 'classical'):
            subset = [r for r in rows if r['model'] == model and (tier is None or r['tier'] == tier)]
            expected = summary['models'][model] if tier is None else summary['by_tier'][tier][model]
            values = dict(episodes=len(subset), clean=sum(r['clean_success'] for r in subset), contact_maps=sum(r['contacts'] > 0 for r in subset))
            if any(expected[k] != v for k, v in values.items()) or abs(expected['clean_rate'] - values['clean'] / values['episodes']) > 1e-12:
                raise ValueError('Saved summary differs from raw reports')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(directory / 'final-rollouts.json') != recorded['final_reports_sha256']:
        raise ValueError('Raw report checksum differs from VERIFICATION.json')
    if digest(directory / 'selection.json') != manifest['selection_sha256']:
        raise ValueError('Selected controller checksum differs from the manifest')
    if summary['selection'] != selection or not selection['selected_before_final']:
        raise ValueError('Selected configuration differs between records')
    checkpoint = ROOT / 'artifacts/imitation/versions/main-23of32.pt'
    if digest(checkpoint) != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum differs from the selected model')
    neural, classical = (summary['models'][m]['clean'] for m in ('neural', 'classical'))
    delta = 100 * (neural - classical) / 96
    if abs(summary['paired']['neural_minus_classical_pp'] - delta) > 1e-10:
        raise ValueError('Incorrect difference in percentage points')
    return dict(scenes=96, reports=192, mercury=neural, reactive=classical, difference_pp=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        result = verify(args.directory)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f'Verification failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
