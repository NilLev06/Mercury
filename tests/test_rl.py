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

        self.assertEqual(state, ("target_right", "clear", "far", "moving"))

    def test_frozen_policy_turns_the_action_with_highest_learned_value_into_motor_command(self):
        agent = QLearningAgent(actions=((0.7, -0.4), (0.7, 0.4)), epsilon=0.0)
        state = ("target_none", "clear", "far", "stalled")
        agent.table[state] = [0.1, 2.0]
        sensor = SensorFrame(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8),
                             0.0, 0.0, 0.8, False, 0.0)

        command, telemetry = QLearningBrain(agent).step(sensor, None, None, 0.0, 0.02, obstacle=None)

        self.assertEqual((command.traction, command.steering), (0.7, 0.4))
        self.assertEqual(telemetry.populations["rl_best_value"], 2.0)


if __name__ == "__main__":
    unittest.main()
