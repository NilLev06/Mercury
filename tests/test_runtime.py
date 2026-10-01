import unittest
from types import SimpleNamespace
import numpy as np
from fly_brain_rc.contracts import SensorFrame, MotionObservation, ActuatorCommand, BrainTelemetry, ObjectObservation, RuntimeTelemetry
from fly_brain_rc.runtime import FlyRuntime, run_episode


class RuntimeTests(unittest.TestCase):
    def test_tick_reads_and_processes_once_and_emits_after_apply(self):
        calls = []
        frame = SensorFrame(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8), 0, 0, 1.2, False, 0)
        command = ActuatorCommand(.2, .1, 0, 1)
        def call(name, value):
            def invoke(*args):
                calls.append(name)
                return value
            return invoke
        env = SimpleNamespace(sensor_frame=call('read', frame), apply=call('apply', None))
        brain = SimpleNamespace(step=call('brain', (command, BrainTelemetry())))
        detector = SimpleNamespace(detect=call('target', None))
        motion = SimpleNamespace(update=call('motion', MotionObservation(0, 0, 0, 0, 0, 0)))
        descriptor = SimpleNamespace(encode=call('descriptor', np.ones(64)))
        memory = SimpleNamespace(observe=call('memory', .3))
        runtime = FlyRuntime(env, brain, .02, detector=detector, motion=motion, descriptor=descriptor, memory=memory, emit=call('emit', None))
        record = runtime.tick()
        self.assertEqual(calls, ['read', 'target', 'motion', 'descriptor', 'memory', 'brain', 'apply', 'emit'])
        self.assertIs(record.sensor, frame)
        self.assertIs(record.command, command)

    def test_zero_steps_rejected(self):
        with self.assertRaises(ValueError):
            run_episode(None, steps=0)

    def test_episode_requires_stable_stop_and_tracks_ground_truth_outside_brain(self):
        from fly_brain_rc.neural.brain import ArrivalTracker
        frame = SensorFrame(np.zeros((8, 8, 3)), np.zeros((8, 8)), 0, 0, 1, False, 0)
        target = ObjectObservation('red_ball', 1, (0, 0, 1, 1), 0, .15, 0)
        motion = MotionObservation(0, 0, 0, 0, 0, 0)
        records = iter(RuntimeTelemetry(i * .02, i + 1, frame, target, motion,
                          BrainTelemetry(familiarity=.25), ActuatorCommand(0, .1 * i, i * .02, i + 1))
                       for i in range(3))
        env = SimpleNamespace(goal_distance=lambda: .2, sim_cfg=SimpleNamespace(goal_radius=.3), collisions=0, penetrations=0)
        runtime = SimpleNamespace(env=env, dt=.02, tick=lambda: next(records))
        result = run_episode(runtime, 3, arrival=ArrivalTracker(duration=.06))
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 3)
        self.assertAlmostEqual(result.steering_variation, .2)
        self.assertEqual(result.acquisition_time, 0)
        self.assertAlmostEqual(result.familiarity_novelty, .75)

    def test_distance_alone_is_not_success(self):
        frame = SensorFrame(np.zeros((8, 8, 3)), np.zeros((8, 8)), 0, 0, 1, False, 0)
        record = RuntimeTelemetry(0, 1, frame, None, MotionObservation(0, 0, 0, 0, 0, 0), BrainTelemetry(), ActuatorCommand(0, 0, 0, 1))
        env = SimpleNamespace(goal_distance=lambda: .1, sim_cfg=SimpleNamespace(goal_radius=.3), collisions=0, penetrations=0)
        result = run_episode(SimpleNamespace(env=env, dt=.02, tick=lambda: record), 2)
        self.assertFalse(result.success)
        self.assertIsNone(result.stable_stop_time)
