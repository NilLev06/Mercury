import unittest
from fly_brain_rc.telemetry import TelemetryBus


class TelemetryTests(unittest.TestCase):
    def test_latest_and_unsubscribe(self):
        bus = TelemetryBus()
        q = bus.subscribe(1)
        bus.publish({'sequence': 1})
        bus.publish({'sequence': 2})
        self.assertEqual(q.get_nowait()['sequence'], 2)
        bus.unsubscribe(q)
        bus.publish({'sequence': 3})
        self.assertTrue(q.empty())

    def test_unbounded_subscriber_rejected(self):
        with self.assertRaises(ValueError):
            TelemetryBus().subscribe(0)

    def test_serialization_omits_raw_frames_and_encodes_preview(self):
        from fly_brain_rc.contracts import SensorFrame, RuntimeTelemetry, BrainTelemetry, MotionObservation, ActuatorCommand
        import numpy as np
        frame = SensorFrame(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8), 0, 0, 1, False, 0)
        record = RuntimeTelemetry(0, 1, frame, None, MotionObservation(0, 0, 0, 0, 0, 0), BrainTelemetry(compass_ring=np.array([1, 2])), ActuatorCommand(0, 0, 0, 1))
        encoded = TelemetryBus().serialize(record)
        self.assertNotIn('rgb', encoded['sensor'])
        self.assertNotIn('gray', encoded['sensor'])
        self.assertTrue(encoded['preview'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(encoded['brain']['compass_ring'], [1, 2])
