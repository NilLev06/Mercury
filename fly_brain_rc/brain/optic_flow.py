import numpy as np
import cv2
from ..config import OpticFlowConfig


class OpticFlowBrain:
    def __init__(self, cfg: OpticFlowConfig):
        self.cfg = cfg
        self.prev = None

    def step(self, frame_gray: np.ndarray) -> dict:
        fmt = cv2.cv2.calcOpticalFlowFarneback if hasattr(cv2, "cv2") else cv2.calcOpticalFlowFarneback
        cfg = self.cfg
        img = cv2.resize(frame_gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img, dtype=np.float32)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if self.prev is None:
            self.prev = img
            return dict(left=0.0, right=0.0, forward=0.0, divergence=0.0)
        flow = fmt(
            self.prev,
            img,
            None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            0,
        )
        self.prev = img
        h, w = flow.shape[:2]
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        cx = w / 2
        col = np.arange(w)
        xx, _ = np.meshgrid(col, np.arange(h))
        side = np.where(xx < cx, 0, 1)
        left_mag = mag[side == 0].mean() if (side == 0).any() else 0.0
        right_mag = mag[side == 1].mean() if (side == 1).any() else 0.0
        band0 = int(w * 0.30)
        band1 = int(w * 0.70)
        forward_mag = mag[:, band0:band1].mean()
        r = np.sqrt((xx - cx) ** 2 + (np.arange(h)[:, None] - h / 2) ** 2) + 1e-6
        div = ((fx * (xx - cx) + fy * (np.arange(h)[:, None] - h / 2)) / r).mean()
        return dict(left=left_mag, right=right_mag, forward=forward_mag, divergence=div)


class H1Steering:
    def __init__(self, gain: float, velocity_gain: float):
        self.gain = gain
        self.velocity_gain = velocity_gain

    def steer(self, fl: dict) -> float:
        bias = self.gain * (fl["left"] - fl["right"]) / max(fl["left"] + fl["right"], 1e-6)
        return float(np.clip(bias, -1.0, 1.0))

    def throttle_factor(self, fl: dict) -> float:
        f = self.velocity_gain * fl["forward"]
        return float(np.clip(1.0 / (1.0 + max(f, 0.0)), 0.35, 1.0))