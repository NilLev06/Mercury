import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from fly_brain_rc.rl import ExplorationReward, QLearningAgent, QLearningBrain, observation_state
from fly_brain_rc.rl_training import observation_state
from fly_brain_rc.rl_training import load_training_agent
from fly_brain_rc.rl_training import train_q_learning
from tests.test_environment import HeadlessRenderer
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.contracts import ObjectObservation, SensorFrame
import numpy as np


class ReinforcementLearningTests(unittest.TestCase):
    def test_training_reads_once_per_tick_and_learns_forward_states(self):
        instances = []
        def make_env(*args, **kwargs):
            env = FlyCarEnv(*args, **kwargs)
            env.sensor_frame = unittest.mock.Mock(wraps=env.sensor_frame)
            instances.append(env)
            return env
        with tempfile.TemporaryDirectory() as directory, \
             patch('fly_brain_rc.sim.env.mujoco.Renderer', HeadlessRenderer), \
             patch('fly_brain_rc.rl_training.FlyCarEnv', side_effect=make_env), \
             patch.object(QLearningAgent, 'choose', return_value=0):
            path = Path(directory) / 'policy.json'
            train_q_learning(episodes=1, steps=20, output=path)
            learner = QLearningAgent.load(path)
            self.assertEqual(instances[0].sensor_frame.call_count, 21)
            self.assertTrue(any('forward' in state for state in learner.table))

    def test_policy_holds_action_despite_changed_observation_until_point_two_seconds(self):
        learner = QLearningAgent(actions=((.3, -.6), (.3, .6)), epsilon=0)
        sensor = SensorFrame(np.zeros((2,2,3), dtype=np.uint8), np.zeros((2,2), dtype=np.uint8),
                             0, 0, .8, False, 0)
        other = replace(sensor, tof=.1, timestamp=.02)
        learner.table[observation_state(sensor, None, None)] = [2, 0]
        learner.table[observation_state(other, None, None)] = [0, 2]
        brain = QLearningBrain(learner)
        self.assertLess(brain.step(sensor, None, None, 0, .02)[0].steering, 0)
        self.assertLess(brain.step(other, None, None, 0, .02)[0].steering, 0)
        self.assertGreater(brain.step(replace(other, timestamp=.2), None, None, 0, .02)[0].steering, 0)

    def test_old_table_is_preserved_when_observation_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            original = '{"version": 2, "table": {"old": [42]}}'
            path.write_text(original)
            agent = load_training_agent(path, seed=0, epsilon=.25)
            self.assertEqual(agent.table, {})
            self.assertEqual(path.with_name('policy.json.v2.bak').read_text(), original)

    def test_new_cell_is_rewarded_and_stationary_streak_is_penalized(self):
        reward = ExplorationReward(cell_size=0.5)
        reward.reset((0.0, 0.0))

        discovered = reward.step((0.6, 0.0), moved=True, contact=False, reached=False)
        first_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)
        repeated_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)

        self.assertGreater(discovered, 0.0)
        self.assertLess(first_idle, 0.0)
        self.assertLess(repeated_idle, first_idle)

    def test_centered_target_gives_more_immediate_motivation_than_no_target(self):
        reward = ExplorationReward()
        reward.reset((0.0, 0.0))
        centered = ObjectObservation("red_ball", 1.0, (.4, .4, .6, .6), 0.0, .05, 0.0)

        no_target = reward.step((0.0, 0.0), moved=True, contact=False, reached=False, target=None)
        target_reward = reward.step((0.0, 0.0), moved=True, contact=False, reached=False, target=centered)

        self.assertGreater(target_reward, no_target)

    def test_stall_penalty_is_capped_and_contact_is_only_charged_on_impact(self):
        reward = ExplorationReward(idle_penalty=.01, max_idle_penalty=.12, contact_penalty=1.5)
        reward.reset((0.0, 0.0))

        tail = [reward.step((0.0, 0.0), moved=False, contact=False, reached=False) for _ in range(200)]
        impact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)
        held_contact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)

        self.assertGreaterEqual(min(tail), -.12)
        self.assertAlmostEqual(impact, -1.5)
        self.assertEqual(held_contact, 0.0)

    def test_positive_experience_increases_selected_action_value(self):
        learner = QLearningAgent(actions=((0.6, 0.0),), alpha=0.5, gamma=0.0, epsilon=0.0)
        state = ("target_center", "clear", "moving")

        learner.update(state, 0, 2.0, state, terminal=True)

        self.assertGreater(learner.value(state, 0), 0.0)

    def test_training_reuses_saved_table_instead_of_starting_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            state = ("target_none", "clear", "far", "moving", "no_contact")
            learner = QLearningAgent(epsilon=0.0)
            learner.update(state, 0, 2.0, state, terminal=True)
            learner.save(path)

            resumed = load_training_agent(path, seed=7, epsilon=.25)

            self.assertGreater(resumed.value(state, 0), 0.0)
            self.assertEqual(resumed.epsilon, .25)
            self.assertEqual(resumed.training_runs, 0)

    def test_policy_state_uses_perception_but_not_world_coordinates(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.002, 0.8, False, 0.0)
        target = ObjectObservation("red_ball", 1.0, (72, 20, 20, 20), 0.4, 0.08, 0.0)

        state = observation_state(sensor, target, obstacle=None)

        self.assertEqual(state, ("target_right", "clear", "small", "clear", "clear", "clear", "stalled", "no_contact"))

    def test_policy_state_distinguishes_the_side_that_needs_an_escape_turn(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.2, True, 0.0, contact_left=True)

        state = observation_state(sensor, target=None, obstacle=None)

        self.assertEqual(state[-1], "contact_left")

    def test_frozen_policy_turns_the_action_with_highest_learned_value_into_motor_command(self):
        agent = QLearningAgent(actions=((0.7, -0.4), (0.7, 0.4)), epsilon=0.0)
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.8, False, 0.0)
        state = observation_state(sensor, None, None)
        agent.table[state] = [0.1, 2.0]

        command, telemetry = QLearningBrain(agent).step(sensor, None, None, 0.0, 0.02, obstacle=None)

        self.assertEqual((command.traction, command.steering), (0.7, 0.4))
        self.assertEqual(telemetry.populations["rl_best_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
