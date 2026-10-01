import unittest
from dataclasses import replace

from fly_brain_rc.contracts import ActuatorCommand
from fly_brain_rc.neural.brain import ArrivalTracker
from tests.test_neural_brain import inputs


class ArrivalTests(unittest.TestCase):
    def test_three_continuous_seconds_and_reset_on_every_violation(self):
        tracker = ArrivalTracker()
        target = inputs(area=0.15)[1]
        command = ActuatorCommand(0, 0, 0, 0)
        for _ in range(149):
            self.assertFalse(tracker.update(target, command, 0, 1, 0.02))
        self.assertTrue(tracker.update(target, command, 0, 1, 0.02))
        self.assertTrue(tracker.arrived)
        violations = [(None, command, 0, 1), (replace(target, bearing=0.5), command, 0, 1),
                      (replace(target, apparent_area=0.001), command, 0, 1),
                      (replace(target, apparent_area=0.8), command, 0, 1),
                      (target, replace(command, traction=-0.5), 0, 1),
                      (target, command, -0.5, 1), (target, command, 0, 0.01)]
        for args in violations:
            tracker.update(target, command, 0, 1, 1)
            self.assertFalse(tracker.update(*args, 0.02))
            self.assertEqual(tracker.elapsed, 0)
        tracker.reset()
        self.assertFalse(tracker.arrived)

    def test_invalid_values_fail_closed(self):
        tracker = ArrivalTracker()
        target = inputs(area=0.15)[1]
        command = ActuatorCommand(0, 0, 0, 0)
        self.assertFalse(tracker.update(target, command, float('nan'), 1, 0.02))
        with self.assertRaises(ValueError):
            tracker.update(target, command, 0, 1, -1)
