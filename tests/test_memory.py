import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fly_brain_rc.memory import FamiliarityMemory, ShortTermMemory


def unit_vector(index: int) -> np.ndarray:
    vector = np.zeros(64, dtype=np.float32)
    vector[index] = 1.0
    return vector


class FamiliarityMemoryTests(unittest.TestCase):
    def test_repeated_view_becomes_familiar(self):
        memory = FamiliarityMemory(capacity=8)
        first = memory.observe(unit_vector(0), now=0.0)
        second = memory.observe(unit_vector(0), now=1.0)
        self.assertEqual(first, 0.0)
        self.assertGreater(second, first)
        self.assertLessEqual(second, 1.0)

    def test_orthogonal_view_does_not_match(self):
        memory = FamiliarityMemory(capacity=8, match_threshold=0.8)
        memory.observe(unit_vector(0), now=0.0)
        self.assertEqual(memory.observe(unit_vector(1), now=1.0), 0.0)
        self.assertEqual(len(memory.entries), 2)

    def test_habituation_decays_with_elapsed_time(self):
        memory = FamiliarityMemory(capacity=8, decay_rate=1.0)
        memory.observe(unit_vector(0), now=0.0)
        recent = memory.observe(unit_vector(0), now=0.0)
        old = memory.observe(unit_vector(0), now=10.0)
        self.assertLess(old, recent)
        self.assertGreater(old, 0.0)

    def test_evicts_least_useful_entry(self):
        memory = FamiliarityMemory(capacity=2, decay_rate=1.0)
        memory.observe(unit_vector(0), now=0.0)
        memory.observe(unit_vector(1), now=1.0)
        memory.observe(unit_vector(1), now=2.0)
        memory.observe(unit_vector(2), now=3.0)
        self.assertEqual(len(memory.entries), 2)
        self.assertEqual(memory.observe(unit_vector(0), now=3.0), 0.0)

    def test_rejects_invalid_descriptors_and_time_without_mutation(self):
        memory = FamiliarityMemory(capacity=2)
        memory.observe(unit_vector(0), now=0.0)
        for descriptor, now in (
            (np.zeros(64), 1.0),
            (np.full(64, np.nan), 1.0),
            (np.ones(3), 1.0),
            (unit_vector(1), np.inf),
        ):
            with self.subTest(descriptor=descriptor.shape, now=now):
                with self.assertRaises(ValueError):
                    memory.observe(descriptor, now)
        self.assertEqual(len(memory.entries), 1)

    def test_persistence_is_bounded_and_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "familiarity.json"
            memory = FamiliarityMemory(capacity=2)
            for i in range(4):
                memory.observe(unit_vector(i), now=float(i))
            self.assertFalse(path.exists())
            memory.save(path)
            loaded = FamiliarityMemory.load(path)
            self.assertLessEqual(len(loaded.entries), 2)
            self.assertEqual(loaded.capacity, 2)
            self.assertGreater(loaded.observe(unit_vector(3), now=4.0), 0.0)

    def test_failed_atomic_replace_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "familiarity.json"
            memory = FamiliarityMemory(capacity=2)
            memory.observe(unit_vector(0), now=0.0)
            memory.save(path)
            previous = path.read_bytes()
            memory.observe(unit_vector(1), now=1.0)
            with patch("fly_brain_rc.memory.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    memory.save(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_load_rejects_corrupt_version_nonfinite_and_oversized_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "familiarity.json"
            memory = FamiliarityMemory(capacity=1)
            memory.observe(unit_vector(0), now=0.0)
            memory.save(path)
            valid = json.loads(path.read_text())
            invalid_payloads = []
            wrong_version = {**valid, "version": 999}
            invalid_payloads.append(wrong_version)
            invalid_payloads.append({**valid, "version": True})
            nonfinite = json.loads(json.dumps(valid))
            nonfinite["entries"][0]["descriptor"][0] = float("nan")
            invalid_payloads.append(nonfinite)
            oversized = json.loads(json.dumps(valid))
            oversized["entries"].append(oversized["entries"][0])
            invalid_payloads.append(oversized)
            malformed = json.loads(json.dumps(valid))
            malformed["entries"][0]["descriptor"] = [1.0, 2.0]
            invalid_payloads.append(malformed)
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        FamiliarityMemory.load(path)


class ShortTermMemoryTests(unittest.TestCase):
    def test_accepts_finite_numpy_scalar_inputs(self):
        memory = ShortTermMemory()
        target, threat = memory.update(np.float32(0.8), np.float64(0.5), np.float32(0.1))
        self.assertGreater(target, 0.0)
        self.assertGreater(threat, 0.0)

    def test_target_and_threat_integrate_decay_and_reset(self):
        memory = ShortTermMemory()
        memory.update(target=1.0, threat=0.8, dt=1.0)
        self.assertGreater(memory.target, 0.0)
        self.assertGreater(memory.threat, 0.0)
        target, threat = memory.target, memory.threat
        memory.update(target=0.0, threat=0.0, dt=1.0)
        self.assertLess(memory.target, target)
        self.assertLess(memory.threat, threat)
        memory.reset()
        self.assertEqual((memory.target, memory.threat), (0.0, 0.0))
        self.assertEqual((ShortTermMemory().target, ShortTermMemory().threat), (0.0, 0.0))

    def test_invalid_update_does_not_change_state(self):
        memory = ShortTermMemory()
        memory.update(1.0, 0.5, 0.1)
        before = memory.target, memory.threat
        for target, threat, dt in ((np.nan, 0.0, 0.1), (0.0, np.inf, 0.1), (0.0, 0.0, 0.0)):
            with self.assertRaises(ValueError):
                memory.update(target, threat, dt)
            self.assertEqual((memory.target, memory.threat), before)


if __name__ == "__main__":
    unittest.main()
