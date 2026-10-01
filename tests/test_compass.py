import math
import unittest

import numpy as np

from fly_brain_rc.neural import CompassState


class CompassTests(unittest.TestCase):
    def test_heading_wraps_across_ring_boundary(self):
        compass = CompassState()
        compass.reset(heading=math.pi - 0.01)
        compass.update(yaw_rate=1.0, optic_rotation=0.0, encoder_delta=0.0, dt=0.1)
        self.assertLess(compass.heading, -3.0)
        self.assertEqual(compass.activity.shape, (72,))
        self.assertAlmostEqual(float(compass.activity.sum()), 1.0)
        self.assertGreater(compass.activity[0], 0.0)
        self.assertGreater(compass.activity[-1], 0.0)

    def test_encoder_travel_accumulates_vector_toward_home(self):
        compass = CompassState()
        compass.update(yaw_rate=0.0, optic_rotation=0.0, encoder_delta=1.0, dt=0.1)
        np.testing.assert_allclose(compass.home_vector, [-1.0, 0.0], atol=1e-10)
        compass.update(yaw_rate=math.pi / 2, optic_rotation=0.0, encoder_delta=2.0, dt=1.0)
        np.testing.assert_allclose(compass.home_vector, [-1.0, -2.0], atol=1e-10)

    def test_optic_correction_is_bounded(self):
        normal = CompassState()
        extreme = CompassState()
        normal.update(yaw_rate=0.0, optic_rotation=1.0, encoder_delta=0.0, dt=1.0)
        extreme.update(yaw_rate=0.0, optic_rotation=1000.0, encoder_delta=0.0, dt=1.0)
        self.assertAlmostEqual(normal.heading, extreme.heading)
        self.assertGreater(normal.heading, 0.0)

    def test_reset_clears_path_integration(self):
        compass = CompassState()
        compass.update(yaw_rate=0.0, optic_rotation=0.0, encoder_delta=1.0, dt=0.1)
        compass.reset(heading=-math.pi / 2)
        np.testing.assert_array_equal(compass.home_vector, [0.0, 0.0])
        self.assertAlmostEqual(compass.heading, -math.pi / 2)

    def test_rejects_nonfinite_sensors_and_invalid_dt(self):
        compass = CompassState()
        for kwargs in (
            dict(yaw_rate=float("nan"), optic_rotation=0.0, encoder_delta=0.0, dt=1.0),
            dict(yaw_rate=0.0, optic_rotation=float("inf"), encoder_delta=0.0, dt=1.0),
            dict(yaw_rate=0.0, optic_rotation=0.0, encoder_delta=float("nan"), dt=1.0),
            dict(yaw_rate=0.0, optic_rotation=0.0, encoder_delta=0.0, dt=0.0),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compass.update(**kwargs)
        np.testing.assert_array_equal(compass.home_vector, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
