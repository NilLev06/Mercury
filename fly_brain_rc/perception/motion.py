"""Continuous optic-flow and image-expansion observations."""

import cv2
import numpy as np

from ..config import OpticFlowConfig
from ..contracts import MotionObservation, SensorFrame


class MotionEstimator:
    def __init__(self, config: OpticFlowConfig | None = None):
        self.config = config or OpticFlowConfig()
        self._previous: np.ndarray | None = None

    def update(self, frame: SensorFrame, dt: float) -> MotionObservation:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        cfg = self.config
        image = cv2.resize(frame.gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image, dtype=np.float32)
        previous = self._previous
        self._previous = image
        if previous is None:
            return MotionObservation(0.0, 0.0, 0.0, 0.0, 0.0, frame.timestamp)

        flow = cv2.calcOpticalFlowFarneback(
            previous,
            image,
            None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            0,
        )
        height, width = flow.shape[:2]
        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.hypot(fx, fy)
        normalizer = max(float(width) * dt, 1e-6)
        left, center, right = np.array_split(magnitude, 3, axis=1)
        regional_flow = [float(np.clip(region.mean() / normalizer, 0.0, 1.0))
                         for region in (left, center, right)]
        optic_rotation = float(np.clip(fx.mean() / normalizer, -1.0, 1.0))

        yy, xx = np.indices((height, width), dtype=np.float32)
        motion_weight = float(magnitude.sum())
        if motion_weight > 1e-6:
            center_x = float((xx * magnitude).sum() / motion_weight)
            center_y = float((yy * magnitude).sum() / motion_weight)
            mean_fx = float((fx * magnitude).sum() / motion_weight)
            mean_fy = float((fy * magnitude).sum() / motion_weight)
            dx = xx - center_x
            dy = yy - center_y
            radial_energy = float(((dx * dx + dy * dy) * magnitude).sum())
            expansion = float((((fx - mean_fx) * dx + (fy - mean_fy) * dy)
                               * magnitude).sum()) / max(radial_energy, 1e-6)
            looming = float(np.clip(expansion / dt, 0.0, 1.0))
        else:
            looming = 0.0

        return MotionObservation(
            flow_left=regional_flow[0],
            flow_center=regional_flow[1],
            flow_right=regional_flow[2],
            optic_rotation=optic_rotation,
            looming=looming,
            timestamp=frame.timestamp,
        )
