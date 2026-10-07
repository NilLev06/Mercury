import unittest
import tempfile
from pathlib import Path

from fly_brain_rc.neural.heading_columns import (
    columns, rank_columns, map_columns, compatible_matches, selected_neurons)


def test_instance_suffix_not_type_number():
    assert columns('EPG(PB08)_L3') == ('L3',)
    assert columns('Delta7(PB15)_L6R3_L') == ('L6', 'R3')
    assert columns('unknown') == ()


def test_aggregate_same_column_before_margin():
    rows = [('1', 'EPG(PB08)_L3', .8),
            ('2', 'EPG(PB08)_L3', .79),
            ('3', 'EPG(PB08)_R4', .6)]
    result = rank_columns(rows)
    assert result['candidate_columns'] == ['L3']
    assert abs(result['margin'] - .2) < 1e-10
    assert result['status'] == 'morphology_supported_hypothesis'


def test_ties_remain_unresolved():
    result = rank_columns([('1', 'EPG(PB08)_L3', .8),
                           ('2', 'EPG(PB08)_R4', .8)])
    assert result['status'] == 'ambiguous'
    assert result['candidate_columns'] is None


def test_invalid_scores_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        rank_columns([('1', 'EPG(PB08)_L3', float('nan'))])


def test_missing_and_multiple_columns():
    assert rank_columns([])['status'] == 'unavailable'
    result = rank_columns([('1', 'Delta7(PB15)_L6R3_L', .8),
                           ('2', 'Delta7(PB15)_L5R4_L', .6)])
    assert result['candidate_columns'] == ['L6', 'R3']
    assert 'angle' not in result


def test_unverified_dataset_rejected_before_optional_import():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'bad.feather'
        path.write_bytes(b'not the official dataset')
        with unittest.TestCase().assertRaisesRegex(ValueError, 'checksum'):
            map_columns(path, path, path)


def test_identifiers_preserved_as_strings():
    result = rank_columns([('720575940626315207', 'EPG(PB08)_L3', .8)])
    assert result['alternatives'][0]['hemibrain_body_id'] == '720575940626315207'


def test_matching_excludes_incompatible_type_and_handles_alias():
    meta = {'1': {'type': 'EPG', 'instance': 'EPG(PB08)_L3'},
            '2': {'type': 'PEN_a(PEN1)', 'instance': 'PEN_a(PB06a)_R4'},
            '4': {'type': 'PEN_a(PEN1)', 'instance': 'PEN_a(PB06a)_L3'}}
    matches = compatible_matches('PEN_a/PEN1', ['1', '2', '3', '4'], [.99, .8, .9, .6], meta)
    assert rank_columns(matches)['candidate_columns'] == ['R4']


def test_duplicate_and_lossy_root_ids_rejected():
    neuron = {'root_id': '720575940626315207', 'cell_type': 'EPG'}
    with unittest.TestCase().assertRaises(ValueError):
        selected_neurons({'neurons': [neuron, neuron]})
    with unittest.TestCase().assertRaises(ValueError):
        selected_neurons({'neurons': [dict(neuron, root_id=720575940626315207)]})


def test_single_alternative_does_not_prove_margin():
    result = rank_columns([('1', 'EPG(PB08)_L3', .9)])
    assert result['candidate_columns'] is None
    assert result['status'] == 'ambiguous'


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in globals().items()
                              if name.startswith('test_'))
