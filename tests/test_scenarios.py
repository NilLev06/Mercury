import unittest

from fly_brain_rc.sim.scenarios import get_scenario, scenario_names


class ScenarioTests(unittest.TestCase):
    def test_navigation_suite_has_distinct_fixed_maps(self):
        self.assertEqual(
            scenario_names(),
            ["blocked_goal", "corridor", "slalom", "turn_corridor"],
        )
        layouts = {
            name: (get_scenario(name).goal, tuple(p for p, _ in get_scenario(name).obstacles))
            for name in scenario_names()
        }
        self.assertEqual(len(set(layouts.values())), 4)

    def test_corridor_is_straight_with_goal_at_far_end(self):
        scenario = get_scenario("corridor")
        self.assertGreater(scenario.goal[0], 2.0)
        self.assertAlmostEqual(scenario.goal[1], 0.0)
        ys = {round(position[1], 2) for position, _ in scenario.obstacles}
        self.assertEqual(ys, {-1.0, 1.0})

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown scenario"):
            get_scenario("missing")


if __name__ == "__main__":
    unittest.main()
