"""Compact grayscale descriptors for later familiarity comparisons."""

import cv2
import numpy as np


class VisualDescriptor:
    def encode(self, gray: np.ndarray) -> np.ndarray:
        image = cv2.resize(
            np.asarray(gray, dtype=np.float32),
            (8, 8),
            interpolation=cv2.INTER_AREA,
        )
        descriptor = image.reshape(64) - float(image.mean())
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1e-8:
            descriptor = np.zeros(64, dtype=np.float32)
            descriptor[0] = 1.0
        else:
            descriptor = descriptor / norm
        return descriptor.astype(np.float32, copy=False)
