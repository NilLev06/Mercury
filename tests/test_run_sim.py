import unittest
import argparse

from scripts.evaluate import acceptance_passes, episode_seeds, suite_required_rate
from unittest.mock import patch
import run


class EvaluationTests(unittest.TestCase):
    def test_dashboard_attach_opens_running_server(self):
        with patch('sys.argv', ['run.py', 'dashboard', '--attach', 'localhost:8123']), patch('webbrowser.open', return_value=True) as opened:
            self.assertEqual(run.main(), 0)
        opened.assert_called_once_with('http://localhost:8123/')

    def test_sim_dashboard_reaches_viewer_parser(self):
        with patch('sys.argv', ['run.py', 'sim', '--dashboard']), patch('run._run_simulator', return_value=0) as launch:
            self.assertEqual(run.main(), 0)
        self.assertTrue(launch.call_args.args[0].dashboard)

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

    def test_train_rl_dispatches_simulation_trainer(self):
        with patch('sys.argv', ['run.py', 'train-rl', '--episodes', '2', '--steps', '9', '--out', 'tmp.json']), \
             patch('run._run_rl_trainer', return_value=0) as train:
            self.assertEqual(run.main(), 0)
        args = train.call_args.args[0]
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.steps, 9)
        self.assertEqual(args.out, 'tmp.json')

    def test_train_rl_uses_mjpython_on_macos_for_camera_rendering(self):
        args = argparse.Namespace(command='train-rl',episodes=2, steps=9, seed=4, epsilon=.2, out='tmp.json', dashboard=False)
        with patch('run.sys.platform', 'darwin'), patch.dict('run.os.environ', {}, clear=True), \
             patch('run._mjpython_path', return_value='/opt/mjpython'), \
             patch('run.os.execv', side_effect=SystemExit(0)) as execute:
            with self.assertRaises(SystemExit):
                run._run_rl_trainer(args)
        self.assertEqual(execute.call_args.args[0], '/opt/mjpython')
        self.assertIn('train-rl', execute.call_args.args[1])
    def test_seed_schedule_is_repeatable_and_contiguous(self):
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])

    def test_acceptance_rejects_any_committed_penetration(self):
        self.assertFalse(acceptance_passes(20, 20, penetrations=1, required_rate=0.75))
        self.assertTrue(acceptance_passes(15, 20, penetrations=0, required_rate=0.75))
        self.assertEqual(episode_seeds(4, start=7), [7, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
