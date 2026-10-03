import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.memory_store import open_memory, save_memory
from fly_brain_rc.perception.target import MockRedBallDetector
import run


class MemoryCliTests(unittest.TestCase):
    def test_double_click_launcher_opens_lab_in_its_project_directory(self):
        import os
        import subprocess
        import sys
        import json
        launcher = Path(__file__).resolve().parents[1] / 'Digital Fly Car.command'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy = root/'Digital Fly Car.command'
            copy.write_bytes(launcher.read_bytes())
            fake = root/'python'
            fake.write_text(f'#!{sys.executable}\nimport os,sys,json\nprint(json.dumps([os.getcwd(),sys.argv[1:]]))\n')
            fake.chmod(0o755)
            result = subprocess.run(['zsh',str(copy)],cwd='/private/tmp',
                                    env=dict(os.environ,DIGITAL_FLY_PYTHON=str(fake)),
                                    capture_output=True,text=True,check=True)
            cwd,args = json.loads(result.stdout)
            self.assertEqual(Path(cwd).resolve(),root.resolve())
            self.assertEqual(args,['run.py','lab'])

    def test_plain_launcher_defaults_to_ephemeral_memory(self):
        with patch('sys.argv', ['run.py', 'sim']), patch('run._run_simulator', return_value=0) as launch:
            self.assertEqual(run.main(), 0)
        self.assertIsNone(launch.call_args.args[0].memory)

    def test_launcher_forwards_memory_and_reset_to_viewer(self):
        with patch('sys.argv', ['run.py', 'sim', '--memory', '/tmp/test-memory.json', '--reset-memory']), patch.dict('os.environ', {'MJPYTHON_BIN': 'test'}), patch('scripts.fly_mode.main') as launch:
            self.assertEqual(run.main(), 0)
            import sys
            self.assertEqual(sys.argv[sys.argv.index('--memory') + 1], '/tmp/test-memory.json')
            self.assertIn('--reset-memory', sys.argv)
        launch.assert_called_once_with()

    def test_roundtrip_and_recoverable_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nested' / 'memory.json'
            memory = open_memory(path)
            memory.observe(np.ones(64), 10.0)
            save_memory(memory, path)
            self.assertEqual(len(open_memory(path).entries), 1)
            self.assertEqual(len(open_memory(path, reset=True).entries), 0)
            self.assertFalse(path.exists())
            self.assertEqual(len(list(path.parent.glob('memory.json.backup-*'))), 1)

    def test_invalid_store_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'memory.json'
            path.write_text('bad')
            with self.assertRaises(ValueError):
                open_memory(path)
            self.assertEqual(path.read_text(), 'bad')

    def test_disabled_persistence_never_reads_or_saves(self):
        with patch('scripts.memory_store.Path') as filesystem, patch('scripts.memory_store.FamiliarityMemory.load') as load, patch('scripts.memory_store.FamiliarityMemory.save') as save:
            memory = open_memory(None)
            memory.observe(np.ones(64), 0.0)
            save_memory(memory, None)
        load.assert_not_called()
        save.assert_not_called()
        filesystem.assert_not_called()

    def test_reset_requires_explicit_memory(self):
        with self.assertRaisesRegex(ValueError, '--reset-memory requires --memory'):
            open_memory(None, reset=True)

    def test_detector_requires_positive_finite_ratio(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MockRedBallDetector(value)


if __name__ == '__main__':
    unittest.main()
