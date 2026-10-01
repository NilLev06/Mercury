import math
import unittest
from fly_brain_rc.sim.suites import training_cases, type_a_cases, type_b_cases


class SuiteTests(unittest.TestCase):
    def test_deterministic_disjoint_suites(self):
        train = training_cases(7, 10)
        self.assertEqual(train, training_cases(7, 10))
        groups = [set(c.identifier for c in cases) for cases in (train, type_a_cases(), type_b_cases())]
        self.assertTrue(all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i)))

    def test_type_a_requires_target_acquisition(self):
        for case in type_a_cases():
            self.assertGreater(abs(math.atan2(case.scenario.goal[1], case.scenario.goal[0])), math.pi / 2)
            self.assertFalse(case.scenario.obstacles)

    def test_type_b_has_open_space_for_a_simple_turn(self):
        for case in type_b_cases():
            self.assertTrue(case.scenario.obstacles)
            self.assertNotEqual(case.scenario.goal[1], 0)
            self.assertLessEqual(len(case.scenario.obstacles), 2)
