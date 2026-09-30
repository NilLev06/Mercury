import numpy as np
import unittest

from fly_brain_rc.brain.goal_vision import GoalDetector


def frame_with_red_box(x0, x1, y0=25, y1=75):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[y0:y1, x0:x1] = (255, 0, 0)
    return frame


class GoalVisionTests(unittest.TestCase):
    def test_blank_frame_has_no_goal(self):
        obs = GoalDetector().step(np.zeros((100, 200, 3), dtype=np.uint8))
        self.assertFalse(obs.visible)


    def test_centered_red_target_has_near_zero_bearing(self):
        obs = GoalDetector().step(frame_with_red_box(80, 120))
        self.assertTrue(obs.visible)
        self.assertLess(abs(obs.bearing), 0.02)
        self.assertGreater(obs.area, 0.09)


    def test_target_bearing_has_image_sign(self):
        detector = GoalDetector()
        left = detector.step(frame_with_red_box(10, 40))
        right = detector.step(frame_with_red_box(160, 190))
        self.assertLess(left.bearing, -0.5)
        self.assertGreater(right.bearing, 0.5)


    def test_tiny_red_noise_is_ignored(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[5:7, 5:7] = (255, 0, 0)
        self.assertFalse(GoalDetector().step(frame).visible)


    def test_largest_red_blob_is_selected(self):
        frame = frame_with_red_box(10, 25, 10, 25)
        frame[25:80, 130:190] = (255, 0, 0)
        obs = GoalDetector().step(frame)
        self.assertTrue(obs.visible)
        self.assertGreater(obs.bearing, 0.5)


if __name__ == "__main__":
    unittest.main()
