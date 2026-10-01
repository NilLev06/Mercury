import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.memory_store import open_memory, save_memory
from fly_brain_rc.perception.target import MockRedBallDetector
import run


class MemoryCliTests(unittest.TestCase):
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

    def test_detector_requires_positive_finite_ratio(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MockRedBallDetector(value)


if __name__ == '__main__':
    unittest.main()
