"""Nonblocking latest-value fanout and browser-safe telemetry encoding."""
import base64
from dataclasses import fields, is_dataclass
from queue import Queue, Empty, Full
from threading import Lock
import time
import numpy as np


class TelemetryBus:
    def __init__(self):
        self._lock = Lock()
        # Encoding is independent of runtime fanout. Slow browser JPEG work
        # must never hold the lock needed by publish/subscribe/unsubscribe.
        self._preview_lock = Lock()
        self._subscribers = set()
        self.latest = None
        self._preview = None
        self._preview_target = None
        self._preview_timestamp = None
        self._preview_time = -float('inf')

    def subscribe(self, maxsize=1):
        if type(maxsize) is not int or maxsize <= 0:
            raise ValueError('maxsize must be positive')
        queue = Queue(maxsize)
        with self._lock:
            self._subscribers.add(queue)
            if self.latest is not None:
                queue.put_nowait(self.latest)
        return queue

    def unsubscribe(self, queue):
        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, record):
        with self._lock:
            self.latest = record
            for queue in self._subscribers:
                while True:
                    try:
                        queue.put_nowait(record)
                        break
                    except Full:
                        try:
                            queue.get_nowait()
                        except Empty:
                            pass

    def serialize(self, record):
        def encode(value):
            if is_dataclass(value):
                return {f.name: encode(getattr(value, f.name)) for f in fields(value)
                        if f.name not in ('rgb', 'gray')}
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value
        result = encode(record)
        if hasattr(record, 'sensor'):
            with self._preview_lock:
                now = time.monotonic()
                if now - self._preview_time >= .1:
                    import cv2
                    image = record.sensor.rgb
                    scale = min(640 / image.shape[1], 480 / image.shape[0], 1)
                    if scale < 1:
                        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
                    ok, output = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if not ok:
                        raise ValueError('cannot encode preview')
                    self._preview = 'data:image/jpeg;base64,' + base64.b64encode(output).decode('ascii')
                    self._preview_target = encode(record.target)
                    self._preview_timestamp = record.timestamp
                    self._preview_time = now
                result['preview'] = self._preview
                result['preview_target'] = self._preview_target
                result['preview_timestamp'] = self._preview_timestamp
        return result
