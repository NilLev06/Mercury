import unittest

from fly_brain_rc.imitation.sensorimotor import select_paths


class SensorimotorGraphTests(unittest.TestCase):
    def test_keeps_directed_sensor_to_output_paths_not_high_degree_dead_ends(self):
        annotations = {str(i): {'cell_type': 'fixture', 'side': 'left'} for i in range(6)}
        graph = select_paths(annotations, [('0','1',3), ('0','1',2),
            ('1','2',7), ('0','3',1000), ('4','0',20), ('2','5',20)],
            {'vision': ['0']}, ['2'], max_hops=2, max_neurons=6)
        self.assertEqual(graph['ids'], ['0','1','2'])
        self.assertEqual(graph['edges'], [[0,1,5], [1,2,7]])
        self.assertEqual(graph['inputs'], {'vision': [0]})
        self.assertEqual(graph['outputs'], [2])

    def test_rejects_disconnected_input_instead_of_inventing_connections(self):
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            select_paths({'0':{}, '1':{}, '2':{}}, [('0','2',5)],
                         {'vision':['0'], 'touch':['1']}, ['2'])

    def test_budget_failure_does_not_silently_crop_a_path(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            select_paths({'0':{}, '1':{}, '2':{}}, [('0','1',5),('1','2',5)],
                         {'vision':['0']}, ['2'], max_neurons=2)

    def test_partial_input_selection_is_explicit_and_records_excluded_roots(self):
        graph = select_paths({'0':{},'1':{},'2':{}}, [('0','2',5)],
            {'vision':['0','1']}, ['2'], require_all_inputs=False)
        self.assertEqual(graph['ids'],['0','2'])
        self.assertEqual(graph['excluded_inputs'],{'vision':['1']})
        with self.assertRaisesRegex(ValueError,'unreachable'):
            select_paths({'0':{},'1':{},'2':{}}, [('0','2',5)],
                {'vision':['0'], 'touch':['1']}, ['2'], require_all_inputs=False)

    def test_rejects_missing_ids_overlap_and_invalid_counts(self):
        for inputs, outputs, edges in [({'v':['9']},['2'],[]),
                ({'v':['0']},['0'],[]), ({'v':['0']},['2'],[('0','2',-1)])]:
            with self.assertRaises(ValueError):
                select_paths({'0':{}, '2':{}},edges,inputs,outputs)
