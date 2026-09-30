import unittest

from scripts.evaluate import acceptance_passes, episode_seeds


class EvaluationTests(unittest.TestCase):
    def test_seed_schedule_is_repeatable_and_contiguous(self):
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])

    def test_acceptance_rejects_any_committed_penetration(self):
        self.assertFalse(acceptance_passes(20, 20, penetrations=1, required_rate=0.75))
        self.assertTrue(acceptance_passes(15, 20, penetrations=0, required_rate=0.75))
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
