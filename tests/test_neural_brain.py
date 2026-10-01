import json
import unittest

import numpy as np

from fly_brain_rc.contracts import SensorFrame, ObjectObservation, MotionObservation
from fly_brain_rc.neural.brain import NeuralFlyBrain
from fly_brain_rc.neural.genome import FlyGenome


def inputs(bearing=0.0, area=0.02, tof=3.0, looming=0.0, contact=False):
    sensor = SensorFrame(np.zeros((2, 2, 3)), np.zeros((2, 2)), 0.0, 0.0, tof, contact, 0.0)
    target = ObjectObservation('red_ball', 1.0, (0, 0, 1, 1), bearing, area, 0.0)
    motion = MotionObservation(0.0, 0.0, 0.0, 0.0, looming, 0.0)
    return sensor, target, motion


class NeuralBrainTests(unittest.TestCase):
    def test_opponent_target_and_threat(self):
        brain = NeuralFlyBrain()
        self.assertFalse(hasattr(brain, 'mode'))
        command, telemetry = brain.step(*inputs(bearing=0.7), 0.0, 0.02)
        self.assertGreater(telemetry.motor_right, telemetry.motor_left)
        self.assertEqual(command.steering, telemetry.motor_right - telemetry.motor_left)
        brain.reset()
        command, telemetry = brain.step(*inputs(tof=0.05, looming=1.0), 0.0, 0.02)
        self.assertLessEqual(command.traction, 0.0)
        self.assertGreater(telemetry.motor_reverse, telemetry.motor_forward)

    def test_reset_reproduces_and_memory_decays(self):
        brain = NeuralFlyBrain()
        first = brain.step(*inputs(bearing=0.7), 0.2, 0.02)
        for _ in range(30):
            brain.step(*inputs(bearing=0.7), 0.2, 0.02)
        old = np.linalg.norm(brain.target_memory)
        sensor, _, motion = inputs()
        for _ in range(200):
            command, telemetry = brain.step(sensor, None, motion, 0.2, 0.02)
        self.assertLess(np.linalg.norm(brain.target_memory), old)
        self.assertEqual(len(telemetry.compass_ring), 72)
        self.assertTrue(all(0 <= value <= 1 for value in telemetry.populations.values()))
        brain.reset()
        self.assertEqual(brain.step(*inputs(bearing=0.7), 0.2, 0.02), first)

    def test_genome_roundtrip_bounds_and_strict_schema(self):
        genome = FlyGenome.default()
        self.assertEqual(FlyGenome.from_json(genome.to_json()), genome)
        self.assertEqual(genome.with_vector(genome.vector()), genome)
        for vector in (np.full_like(genome.vector(), np.nan), np.full_like(genome.vector(), -1), [1]):
            with self.assertRaises(ValueError):
                genome.with_vector(vector)
        payload = json.loads(genome.to_json())
        for invalid in ({**payload, 'version': 99}, {'version': 1}, {**payload, 'extra': 1}):
            with self.assertRaises(ValueError):
                FlyGenome.from_json(json.dumps(invalid))

    def test_invalid_step_does_not_change_state(self):
        brain = NeuralFlyBrain()
        for familiarity, dt in ((float('nan'), 0.02), (0, 0), (2, 0.02)):
            with self.assertRaises(ValueError):
                brain.step(*inputs(), familiarity, dt)
        self.assertEqual(brain.step(*inputs(), 0, 0.02), NeuralFlyBrain().step(*inputs(), 0, 0.02))

    def test_simultaneous_first_step_does_not_reuse_new_threat(self):
        _, telemetry = NeuralFlyBrain().step(*inputs(looming=1.0, area=0.0), 0, 0.02)
        # Both drives equal 8 with zero previous threat. Sequential reuse
        # would add newly computed threat to reverse and break equality.
        self.assertAlmostEqual(telemetry.threat, telemetry.motor_reverse, places=14)
        self.assertAlmostEqual(telemetry.threat, 0.1666107749782556, places=12)

    def test_centered_arrival_size_stops_and_contact_reverses(self):
        brain = NeuralFlyBrain()
        for _ in range(200):
            command, _ = brain.step(*inputs(area=0.15), 0, 0.02)
        self.assertLess(abs(command.traction), 0.05)
        self.assertLess(abs(command.steering), 0.05)
        command, _ = brain.step(*inputs(area=0.15, contact=True), 0, 0.02)
        self.assertLess(command.traction, 0)

    def test_familiarity_inhibits_exploration_and_missing_target_searches(self):
        unfamiliar, familiar = NeuralFlyBrain(), NeuralFlyBrain()
        sensor, _, motion = inputs()
        for _ in range(100):
            command, novel = unfamiliar.step(sensor, None, motion, 0, 0.02)
            _, known = familiar.step(sensor, None, motion, 1, 0.02)
        self.assertGreater(novel.populations['exploration'], known.populations['exploration'])
        self.assertGreater(command.traction, 0)
        self.assertNotEqual(command.steering, 0)
