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
