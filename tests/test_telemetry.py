import unittest
from fly_brain_rc.telemetry import TelemetryBus


class TelemetryTests(unittest.TestCase):
    def test_publish_does_not_wait_for_stalled_preview_encoder(self):
        from fly_brain_rc.contracts import SensorFrame, RuntimeTelemetry, BrainTelemetry, MotionObservation, ActuatorCommand
        from threading import Thread, Event
        from unittest.mock import patch
        import cv2
        import numpy as np
        frame = SensorFrame(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8), 0, 0, 1, False, 0)
        record = RuntimeTelemetry(0, 1, frame, None, MotionObservation(0, 0, 0, 0, 0, 0), BrainTelemetry(), ActuatorCommand(0, 0, 0, 1))
        bus = TelemetryBus()
        queue = bus.subscribe(1)
        entered, release, published = Event(), Event(), Event()
        original_encoder = cv2.imencode
        def stalled_encoder(*args):
            entered.set()
            release.wait(3)
            return original_encoder(*args)
        def publish():
            bus.publish({'sequence': 2})
            published.set()
        with patch('cv2.imencode', side_effect=stalled_encoder):
            serializer = Thread(target=bus.serialize, args=(record,))
            publisher = Thread(target=publish)
            serializer.start()
            try:
                self.assertTrue(entered.wait(1))
                publisher.start()
                self.assertTrue(published.wait(.5), 'publish waited for preview encoding')
                self.assertEqual(queue.get_nowait()['sequence'], 2)
            finally:
                release.set()
                serializer.join(3)
                if publisher.ident is not None:
                    publisher.join(3)

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
