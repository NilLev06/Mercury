import unittest
import inspect

import numpy as np

from scripts.run_sim import build_brain


def blank_inputs():
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    gray = np.zeros((96, 128), dtype=np.float32)
    ranges = {"front_left": 1000.0, "center": 1000.0, "front_right": 1000.0}
    return rgb, gray, ranges


class ControllerTests(unittest.TestCase):
    def test_controller_api_has_no_ground_truth_goal_inputs(self):
        parameters = inspect.signature(type(build_brain()).step).parameters
        self.assertNotIn("goal_bearing_deg", parameters)
        self.assertNotIn("dist_to_goal", parameters)

    def test_visible_goal_drives_steering_toward_image_bearing(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[30:70, 90:120] = (255, 0, 0)

        throttle, steer = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "approach")
        self.assertGreater(throttle, 0.0)

        self.assertGreater(steer, 0.2)

    def test_search_is_deterministic_without_goal(self):
        args = blank_inputs()
        first = build_brain().step(*args, yaw_rate_deg=0.0, dt=0.02)
        second = build_brain().step(*args, yaw_rate_deg=0.0, dt=0.02)
        self.assertEqual(first, second)

    def test_close_obstacle_has_priority_over_visible_goal(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[30:70, 90:120] = (255, 0, 0)
        ranges["center"] = 0.1

        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "avoid")
        self.assertLess(throttle, 0.0)

    def test_solid_contact_forces_reverse_escape(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        ranges["contact"] = True

        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "avoid")
        self.assertLess(throttle, 0.0)

    def test_range_returns_from_visible_goal_do_not_trigger_retreat(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[38:58, 54:74] = (255, 0, 0)
        ranges.update(front_left=0.02, center=0.42, front_right=0.02)

        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "approach")
        self.assertGreater(throttle, 0.0)

    def test_center_obstacle_is_avoided_even_when_goal_is_visible(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[30:70, 54:74] = (255, 0, 0)
        ranges.update(front_left=1.0, center=0.1, front_right=1.0)

        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "avoid")
        self.assertLess(throttle, 0.0)

    def test_large_visible_goal_does_not_hide_isolated_center_obstacle(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[12:84, 34:94] = (255, 0, 0)
        ranges.update(front_left=1.0, center=0.05, front_right=1.0)

        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "avoid")
        self.assertLess(throttle, 0.0)

    def test_close_goal_memory_does_not_cause_retreat_when_it_leaves_frame(self):
        brain = build_brain()
        rgb, gray, ranges = blank_inputs()
        rgb[12:84, 34:94] = (255, 0, 0)
        ranges.update(front_left=1.0, center=0.18, front_right=1.0)
        throttle, _ = brain.step(rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)
        self.assertGreater(throttle, 0.0)

        blank, gray, ranges = blank_inputs()
        ranges.update(front_left=1.0, center=0.17, front_right=0.34)
        throttle, _ = brain.step(blank, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "approach")
        self.assertGreater(throttle, 0.0)

        close_rgb, gray, ranges = blank_inputs()
        close_rgb[24:72, 52:76] = (255, 0, 0)
        ranges.update(front_left=0.03, center=0.01, front_right=0.03)
        throttle, _ = brain.step(close_rgb, gray, ranges, yaw_rate_deg=0.0, dt=0.02)

        self.assertEqual(brain.mode, "approach")
        self.assertGreater(throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
