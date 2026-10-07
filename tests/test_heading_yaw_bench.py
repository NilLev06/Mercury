import unittest
import numpy as np


class HeadingYawBenchTests(unittest.TestCase):
    def test_stationary_compass_cannot_pass_by_matching_full_turn_endpoint(self):
        from fly_brain_rc.neural.heading_yaw_bench import trajectory_scores
        expected=np.linspace(0,2*np.pi,201)[:,None]
        scores=trajectory_scores(np.zeros_like(expected),expected,np.full_like(expected,.5),.01)
        self.assertAlmostEqual(scores['mean_final_error_degrees'],0.,places=6)
        self.assertAlmostEqual(scores['maximum_error_degrees'],180.,places=6)
        self.assertAlmostEqual(scores['speed_rmse_rad_s'],np.pi,places=6)
        self.assertFalse(scores['passed'])

    def test_wraparound_is_not_a_heading_jump(self):
        from fly_brain_rc.neural.heading_yaw_bench import trajectory_scores
        reference=np.array([[3.13],[3.15],[3.17]])
        actual=np.arctan2(np.sin(reference),np.cos(reference))
        scores=trajectory_scores(actual,reference,np.full_like(actual,.5),.01)
        self.assertTrue(scores['passed'])
        self.assertAlmostEqual(scores['speed_rmse_rad_s'],0.,places=6)
        self.assertLess(scores['maximum_step_degrees'],1.2)

    def test_missing_bump_is_failure_even_if_decoded_angle_matches(self):
        from fly_brain_rc.neural.heading_yaw_bench import trajectory_scores
        angle=np.zeros((3,1))
        scores=trajectory_scores(angle,angle,np.zeros_like(angle),.01)
        self.assertFalse(scores['passed'])
        self.assertEqual(scores['invalid_samples'],3)

    def test_pinned_states_fail_sensitivity_even_with_small_absolute_error(self):
        from fly_brain_rc.neural.heading_yaw_bench import sensitivity_scores
        angles=np.linspace(-np.pi,np.pi,12,endpoint=False)
        scores=sensitivity_scores(angles,angles,np.ones((2,12))*.5,.01)
        self.assertFalse(scores['passed'])
        self.assertEqual(scores['collapsed_pairs'],12)
        self.assertAlmostEqual(scores['median_sensitivity'],0.)

    def test_continuous_memory_preserves_small_offsets_across_wraparound(self):
        from fly_brain_rc.neural.heading_yaw_bench import sensitivity_scores
        angles=np.array([-np.pi,0.,np.pi-.001])
        wrap=lambda a:np.arctan2(np.sin(a),np.cos(a))
        scores=sensitivity_scores(wrap(angles-.01),wrap(angles+.01),np.ones((2,3))*.5,.01)
        self.assertTrue(scores['passed'])
        self.assertEqual(scores['collapsed_pairs'],0)
        self.assertAlmostEqual(scores['median_sensitivity'],1.)

    def test_sensitivity_replay_matches_numpy_neural_dynamics(self):
        from tests.test_heading_yaw import yaw_fixture
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading
        from fly_brain_rc.neural.heading_yaw_bench import evaluate_sensitivity,sensitivity_scores
        a,p,parameters=yaw_fixture();angles=[];contrasts=[]
        for phase in (.09,.19,.11,.21):
            model=AnatomicalHeading(a,p,parameters)
            for _ in range(100):model.step(.01,cue_phase=phase)
            for _ in range(2):model.step(.01)
            angles.append(model.heading if model.heading is not None else 0.)
            contrasts.append(model.contrast)
        expected=sensitivity_scores(angles[:2],angles[2:],np.asarray(contrasts).reshape(2,2),.01)
        actual=evaluate_sensitivity(a,p,parameters,seconds=.02,directions=(.1,.2))
        self.assertEqual(actual['pairs'],2)
        self.assertEqual(actual['passed'],expected['passed'])
        self.assertAlmostEqual(actual['median_sensitivity'],expected['median_sensitivity'],places=4)
        self.assertFalse(actual['vehicle_integration_allowed'])
