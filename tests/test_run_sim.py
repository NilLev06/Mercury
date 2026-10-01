import unittest

from scripts.evaluate import acceptance_passes, episode_seeds, suite_required_rate
from unittest.mock import patch
import run


class EvaluationTests(unittest.TestCase):
    def test_suite_rates_and_explicit_override(self):
        self.assertEqual(suite_required_rate('type-a'), .9)
        self.assertEqual(suite_required_rate('type-b'), .7)
        self.assertEqual(suite_required_rate('type-a', .8), .8)

    def test_acceptance_rejects_collision_even_without_penetration(self):
        self.assertFalse(acceptance_passes(10, 10, penetrations=0, required_rate=.9, collisions=1))

    def test_launcher_defaults_fit_held_out_suite_and_preserve_rate_selection(self):
        with patch('sys.argv', ['run.py', 'evaluate']), patch('run.subprocess.call', return_value=0) as call:
            self.assertEqual(run.main(), 0)
        command = call.call_args.args[0]
        self.assertEqual(command[command.index('--episodes') + 1], '10')
        self.assertNotIn('--required-rate', command)
    def test_seed_schedule_is_repeatable_and_contiguous(self):
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])

    def test_acceptance_rejects_any_committed_penetration(self):
        self.assertFalse(acceptance_passes(20, 20, penetrations=1, required_rate=0.75))
        self.assertTrue(acceptance_passes(15, 20, penetrations=0, required_rate=0.75))
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
