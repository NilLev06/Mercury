import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_gallery_launcher_uses_local_virtual_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / 'Fly Map Gallery.command'
            launcher.write_bytes((ROOT / launcher.name).read_bytes())
            python = root / '.venv/bin/python'
            python.parent.mkdir(parents=True)
            python.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ('DIGITAL_FLY_PYTHON', 'MERCURY_PYTHON')}
            run = subprocess.run(['zsh', str(launcher)], cwd='/private/tmp', env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), ['scripts/run_map_gallery.py'])

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS viewer launcher')
    def test_viewer_launcher_uses_mjpython_from_local_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / 'Try Complex Fly.command'
            launcher.write_bytes((ROOT / launcher.name).read_bytes())
            for name in ['python', 'mjpython']:
                executable = root / '.venv/bin' / name
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_text(f'#!{sys.executable}\nprint({name!r})\n')
                executable.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ('DIGITAL_FLY_PYTHON', 'MERCURY_PYTHON')}
            run = subprocess.run(['zsh', str(launcher)], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), 'mjpython')

    def test_saved_results_reject_a_missing_partner(self):
        script = ROOT / 'scripts/verify_saved_results.py'
        self.assertTrue(script.exists(), 'Saved-results verifier is missing')
        source = ROOT / 'results/2026-10-14-classical-comparison'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ['summary.json', 'final-manifest.json', 'final-rollouts.json', 'selection.json', 'VERIFICATION.json']:
                (target / name).write_bytes((source / name).read_bytes())
            rows = json.loads((target / 'final-rollouts.json').read_text())
            (target / 'final-rollouts.json').write_text(json.dumps(rows[:-1]))
            run = subprocess.run([sys.executable, str(script), '--directory', str(target)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('192', run.stderr)


if __name__ == '__main__':
    unittest.main()
