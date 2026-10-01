import unittest

from fly_brain_rc.brain.reflex import BristleReflex
from fly_brain_rc.config import ReflexConfig


class ReflexTests(unittest.TestCase):
    def test_left_obstacle_commands_right_turn(self):
        reflex = BristleReflex(ReflexConfig())
        result = reflex.step(
            {"front_left": 0.1, "center": 1.0, "front_right": 1.0}, 0.02
        )
        self.assertLess(result["steer"], 0.0)

    def test_right_obstacle_commands_left_turn(self):
        reflex = BristleReflex(ReflexConfig())
        result = reflex.step(
            {"front_left": 1.0, "center": 1.0, "front_right": 0.1}, 0.02
        )
        self.assertGreater(result["steer"], 0.0)


if __name__ == "__main__":
    unittest.main()
