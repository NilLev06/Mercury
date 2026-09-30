from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GoalObservation:
    visible: bool
    bearing: float = 0.0
    area: float = 0.0
    confidence: float = 0.0


class GoalDetector:
    def __init__(self, min_area_ratio: float = 0.001):
        self.min_area_ratio = min_area_ratio

    def step(self, frame_rgb: np.ndarray) -> GoalObservation:
        if frame_rgb.size == 0:
            return GoalObservation(False)
        rgb = np.ascontiguousarray(frame_rgb[..., :3], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        low_red = cv2.inRange(hsv, (0, 120, 80), (12, 255, 255))
        high_red = cv2.inRange(hsv, (168, 120, 80), (179, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GoalObservation(False)

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        frame_area = float(rgb.shape[0] * rgb.shape[1])
        area = contour_area / frame_area
        if area < self.min_area_ratio:
            return GoalObservation(False)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return GoalObservation(False)
        center_x = moments["m10"] / moments["m00"]
        bearing = (center_x / max(rgb.shape[1] - 1, 1)) * 2.0 - 1.0
        x, y, w, h = cv2.boundingRect(contour)
        fill = contour_area / max(float(w * h), 1.0)
        confidence = min(1.0, fill * min(1.0, area / self.min_area_ratio))
        return GoalObservation(True, float(bearing), area, float(confidence))
