import unittest

from fly_brain_rc.rl import ExplorationReward, QLearningAgent, QLearningBrain
from fly_brain_rc.rl_training import observation_state
from fly_brain_rc.contracts import ObjectObservation, SensorFrame
import numpy as np


class ReinforcementLearningTests(unittest.TestCase):
    def test_new_cell_is_rewarded_and_stationary_streak_is_penalized(self):
        reward = ExplorationReward(cell_size=0.5)
        reward.reset((0.0, 0.0))

        discovered = reward.step((0.6, 0.0), moved=True, contact=False, reached=False)
        first_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)
        repeated_idle = reward.step((0.6, 0.0), moved=False, contact=False, reached=False)

        self.assertGreater(discovered, 0.0)
        self.assertLess(first_idle, 0.0)
        self.assertLess(repeated_idle, first_idle)

    def test_stall_penalty_is_capped_and_contact_is_only_charged_on_impact(self):
        reward = ExplorationReward(idle_penalty=.01, max_idle_penalty=.12, contact_penalty=1.5,
                                   max_stall_steps=3)
        reward.reset((0.0, 0.0))

        tail = [reward.step((0.0, 0.0), moved=False, contact=False, reached=False) for _ in range(200)]
        impact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)
        held_contact = reward.step((0.0, 0.0), moved=True, contact=True, reached=False)

        self.assertGreaterEqual(min(tail), -.12)
        self.assertAlmostEqual(impact, -1.5)
        self.assertEqual(held_contact, 0.0)
        reward.reset((0.0, 0.0))
        for _ in range(3):
            reward.step((0.0, 0.0), moved=False, contact=False, reached=False)
        self.assertTrue(reward.is_trapped)

    def test_positive_experience_increases_selected_action_value(self):
        learner = QLearningAgent(actions=((0.6, 0.0),), alpha=0.5, gamma=0.0, epsilon=0.0)
        state = ("target_center", "clear", "moving")

        learner.update(state, 0, 2.0, state, terminal=True)

        self.assertGreater(learner.value(state, 0), 0.0)

    def test_policy_state_uses_perception_but_not_world_coordinates(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.002, 0.8, False, 0.0)
        target = ObjectObservation("red_ball", 1.0, (72, 20, 20, 20), 0.4, 0.08, 0.0)

        state = observation_state(sensor, target, obstacle=None)

        self.assertEqual(state, ("target_right", "clear", "far", "moving", "no_contact"))

    def test_policy_state_distinguishes_the_side_that_needs_an_escape_turn(self):
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.2, True, 0.0, contact_left=True)

        state = observation_state(sensor, target=None, obstacle=None)

        self.assertEqual(state[-1], "contact_left")

    def test_frozen_policy_turns_the_action_with_highest_learned_value_into_motor_command(self):
        agent = QLearningAgent(actions=((0.7, -0.4), (0.7, 0.4)), epsilon=0.0)
        state = ("target_none", "clear", "far", "stalled", "no_contact")
        agent.table[state] = [0.1, 2.0]
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.8, False, 0.0)

        command, telemetry = QLearningBrain(agent).step(sensor, None, None, 0.0, 0.02, obstacle=None)

        self.assertEqual((command.traction, command.steering), (0.7, 0.4))
        self.assertEqual(telemetry.populations["rl_best_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
