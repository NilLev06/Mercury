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


if __name__ == "__main__":
    unittest.main()
