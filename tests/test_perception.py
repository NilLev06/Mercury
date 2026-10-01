import unittest

import cv2
import numpy as np

from fly_brain_rc.contracts import MotionObservation, ObjectObservation, SensorFrame
from fly_brain_rc.perception import MockRedBallDetector, MotionEstimator, VisualDescriptor


def make_frame(rgb: np.ndarray, timestamp: float = 4.2) -> SensorFrame:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return SensorFrame(
        rgb=rgb,
        gray=gray,
        yaw_rate=0.0,
        encoder_delta=0.0,
        tof=1.0,
        contact=False,
        timestamp=timestamp,
    )


def red_frame() -> SensorFrame:
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.circle(rgb, (96, 48), 14, (255, 0, 0), -1)
    return make_frame(rgb)


def square_frame(center_x: int, size: int, timestamp: float) -> SensorFrame:
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    half = size // 2
    rgb[48 - half:48 + half, center_x - half:center_x + half] = 255
    return make_frame(rgb, timestamp)


class PerceptionTests(unittest.TestCase):
    def test_red_ball_detector_emits_backend_neutral_observation(self):
        obs = MockRedBallDetector().detect(red_frame(), "red_ball")
        self.assertIsInstance(obs, ObjectObservation)
        self.assertEqual(obs.class_name, "red_ball")
        self.assertGreater(obs.bearing, 0.0)
        self.assertGreater(obs.apparent_area, 0.0)
        self.assertEqual(obs.timestamp, 4.2)
        self.assertGreater(obs.confidence, 0.0)
        self.assertLessEqual(obs.confidence, 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in obs.bbox))
        self.assertLess(obs.bbox[0], obs.bbox[2])
        self.assertLess(obs.bbox[1], obs.bbox[3])

    def test_detector_rejects_unknown_target_class(self):
        self.assertIsNone(MockRedBallDetector().detect(red_frame(), "bottle"))

    def test_detector_ignores_blank_frame_and_small_noise(self):
        detector = MockRedBallDetector()
        blank = np.zeros((96, 128, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(make_frame(blank), "red_ball"))
        blank[4:6, 4:6] = (255, 0, 0)
        self.assertIsNone(detector.detect(make_frame(blank), "red_ball"))

    def test_descriptor_is_fixed_size_and_normalized(self):
        descriptor = VisualDescriptor().encode(np.ones((96, 128), np.float32))
        self.assertEqual(descriptor.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)
        self.assertTrue(np.isfinite(descriptor).all())

    def test_descriptor_keeps_spatial_structure_after_mean_removal(self):
        gray = np.zeros((8, 8), dtype=np.float32)
        gray[:, 4:] = 255.0
        descriptor = VisualDescriptor().encode(gray)
        self.assertLess(float(descriptor[0]), 0.0)
        self.assertGreater(float(descriptor[7]), 0.0)
        self.assertAlmostEqual(float(descriptor.sum()), 0.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)

    def test_motion_first_frame_is_zero_and_stamped(self):
        obs = MotionEstimator().update(square_frame(64, 24, 1.5), dt=0.1)
        self.assertIsInstance(obs, MotionObservation)
        self.assertEqual(obs.timestamp, 1.5)
        self.assertEqual(
            (obs.flow_left, obs.flow_center, obs.flow_right, obs.optic_rotation, obs.looming),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )

    def test_rightward_translation_changes_lateral_flow(self):
        estimator = MotionEstimator()
        estimator.update(square_frame(86, 24, 0.0), dt=0.1)
        obs = estimator.update(square_frame(96, 24, 0.1), dt=0.1)
        self.assertGreater(obs.flow_right, obs.flow_left)
        self.assertGreater(obs.optic_rotation, 0.0)
        for flow in (obs.flow_left, obs.flow_center, obs.flow_right):
            self.assertGreaterEqual(flow, 0.0)
            self.assertLessEqual(flow, 1.0)

    def test_expanding_square_increases_looming_above_static_pair(self):
        static = MotionEstimator()
        static.update(square_frame(64, 24, 0.0), dt=0.1)
        baseline = static.update(square_frame(64, 24, 0.1), dt=0.1)

        expanding = MotionEstimator()
        expanding.update(square_frame(64, 24, 0.0), dt=0.1)
        expanded = expanding.update(square_frame(64, 36, 0.1), dt=0.1)
        self.assertGreater(expanded.looming, baseline.looming)
        self.assertGreaterEqual(expanded.looming, 0.0)
        self.assertLessEqual(expanded.looming, 1.0)

    def test_translation_does_not_look_like_expansion(self):
        estimator = MotionEstimator()
        estimator.update(square_frame(86, 24, 0.0), dt=0.1)
        translated = estimator.update(square_frame(96, 24, 0.1), dt=0.1)
        self.assertLess(translated.looming, 0.1)


if __name__ == "__main__":
    unittest.main()
