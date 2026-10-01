import math
import unittest

import numpy as np

from fly_brain_rc.neural import RateLayer


class RateLayerTests(unittest.TestCase):
    def test_every_neuron_uses_same_previous_vector(self):
        layer = RateLayer(tau=1.0)
        previous = np.array([1.0, 0.0])
        recurrent = np.array([[0.0, 1.0], [1.0, 0.0]])

        result = layer.step(previous, recurrent, np.zeros(2), dt=0.1)

        np.testing.assert_allclose(result, [0.95, 0.1 / (1.0 + math.exp(-1.0))])
        np.testing.assert_array_equal(previous, [1.0, 0.0])

    def test_external_input_and_bias_drive_individual_neurons(self):
        layer = RateLayer(tau=np.array([1.0, 2.0]), bias=np.array([0.0, -1.0]))
        result = layer.step(np.zeros(2), np.zeros((2, 2)), np.array([0.0, 1.0]), dt=1.0)
        np.testing.assert_allclose(result, [0.5, 0.25])

    def test_extreme_drive_stays_finite_and_activity_is_clipped(self):
        layer = RateLayer(tau=0.01)
        result = layer.step(np.array([0.0, 1.0]), np.zeros((2, 2)),
                            np.array([1e6, -1e6]), dt=1.0)
        np.testing.assert_array_equal(result, [1.0, 0.0])
        self.assertTrue(np.isfinite(result).all())

    def test_rejects_invalid_time_constants_and_input_shapes(self):
        for tau in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                RateLayer(tau=tau)
        layer = RateLayer()
        cases = (
            (np.zeros((2, 1)), np.zeros((2, 2)), np.zeros(2)),
            (np.zeros(2), np.zeros((2, 3)), np.zeros(2)),
            (np.zeros(2), np.zeros((2, 2)), np.zeros(3)),
            (np.zeros(2), np.zeros((2, 2)), np.array([0.0, float("nan")])),
        )
        for previous, recurrent, external in cases:
            with self.subTest(shapes=(previous.shape, recurrent.shape, external.shape)), self.assertRaises(ValueError):
                layer.step(previous, recurrent, external, dt=0.1)
        with self.assertRaises(ValueError):
            layer.step(np.zeros(2), np.zeros((2, 2)), np.zeros(2), dt=0.0)


if __name__ == "__main__":
    unittest.main()
