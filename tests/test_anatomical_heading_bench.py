import unittest

from tests.test_anatomical_heading import fixture


class AnatomicalHeadingBenchTests(unittest.TestCase):
    def test_zero_gain_fails_memory_and_never_claims_integration(self):
        from fly_brain_rc.neural.anatomical_heading import pilot_parameters
        from fly_brain_rc.neural.anatomical_heading_bench import evaluate
        a,p=fixture()
        result=evaluate(a,p,pilot_parameters(excitation=0,delta7_gain=0),
                        cue_seconds=.2,dark_seconds=1.,directions=(.2,1.2))
        self.assertFalse(result['memory_gate_passed'])
        self.assertFalse(result['yaw_integration_implemented'])
        self.assertEqual(result['synapses'],12)
        self.assertEqual(result['regional_rows'],3)
        self.assertEqual(len(result['cases']),2)
        self.assertTrue(all(c['dark_heading'] is None for c in result['cases']))
        for case in result['cases']:
            self.assertIn('maximum_dark_error_degrees',case)
            self.assertGreaterEqual(case['maximum_dark_error_degrees'],case['dark_error_degrees'])

    def test_grid_keeps_all_candidates_and_parameter_assumptions(self):
        from fly_brain_rc.neural.anatomical_heading_bench import sweep
        a,p=fixture()
        result=sweep(a,p,excitations=(0.,1.),delta7_gains=(-1.,1.),
                     cue_seconds=.1,dark_seconds=.2,directions=(.2,))
        self.assertEqual(len(result['candidates']),4)
        self.assertFalse(result['vehicle_integration_allowed'])
        self.assertIn(result['selected_diagnostic_index'],range(4))
        self.assertTrue(all(c['parameters']['gain']['EPG']['Delta7'] in (-1.,1.)
                            for c in result['candidates']))
