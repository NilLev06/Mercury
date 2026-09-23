import numpy as np
import cv2
from ..config import LoomingConfig


class LoomingDetector:
    def __init__(self, cfg: LoomingConfig):
        self.cfg = cfg
        self.prev = None
        self.cooldown = 0.0
        self._prev_area = None

    def step(self, frame_gray: np.ndarray, dt: float) -> dict:
        img = cv2.resize(frame_gray, (64, 48), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img, dtype=np.float32)
        if self.prev is None:
            self.prev = img
            return dict(alarm=False, side=0.0, tau=1e9)
        diff = np.abs(img - self.prev)
        self.prev = img
        moving = diff > 6.0
        area = float(moving.mean())
        cy, cx = np.indices(moving.shape)
        if area > 0.002 and moving.any():
            cen_x = float(cx[moving].mean() / moving.shape[1])
        else:
            cen_x = 0.5
        prev_a = self._prev_area
        if prev_a is not None and prev_a > 0 and area > 0.002:
            rate = (area - prev_a) / (prev_a * dt)
            tau = 1.0 / max(rate, 1e-9)
        else:
            tau = 1e9
        self._prev_area = area if area > 0.002 else None
        self.cooldown = max(0.0, self.cooldown - dt)
        alarm = self.cooldown <= 0 and tau < self.cfg.tau_threshold and area > 0.002
        if alarm:
            self.cooldown = self.cfg.cooldown
        return dict(alarm=bool(alarm), side=(cen_x - 0.5) * 2.0, tau=float(tau))