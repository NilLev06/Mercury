"""Target detection interface and the simulator's red-ball backend."""

from typing import Protocol
import math

import cv2
import numpy as np

from ..contracts import ObjectObservation, SensorFrame


class TargetDetector(Protocol):
    def detect(self, frame: SensorFrame, target_class: str) -> ObjectObservation | None:
        """Observe one target class in a sensor frame."""


class MockRedBallDetector:
    """HSV detector for the simulator's red ball, isolated from the brain."""

    def __init__(self, min_area_ratio: float = 0.001):
        if (isinstance(min_area_ratio, bool)
                or not isinstance(min_area_ratio, (int, float))
                or not math.isfinite(min_area_ratio) or min_area_ratio <= 0):
            raise ValueError('min_area_ratio must be finite and positive')
        self.min_area_ratio = min_area_ratio

    def detect(self, frame: SensorFrame, target_class: str) -> ObjectObservation | None:
        if target_class != "red_ball" or frame.rgb.size == 0:
            return None

        rgb = np.ascontiguousarray(frame.rgb[..., :3], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        low_red = cv2.inRange(hsv, (0, 120, 80), (12, 255, 255))
        high_red = cv2.inRange(hsv, (168, 120, 80), (179, 255, 255))
        mask = cv2.bitwise_or(low_red, high_red)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        height, width = rgb.shape[:2]
        area = contour_area / float(height * width)
        if area < self.min_area_ratio:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None

        center_x = moments["m10"] / moments["m00"]
        bearing = center_x / max(width - 1, 1) * 2.0 - 1.0
        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
        fill = contour_area / float(box_width * box_height)
        confidence = min(1.0, fill * min(1.0, area / self.min_area_ratio))
        return ObjectObservation(
            class_name="red_ball",
            confidence=float(confidence),
            bbox=tuple(float(value) for value in bbox),
            bearing=float(bearing),
            apparent_area=area,
            timestamp=frame.timestamp,
        )


class MockBlueObstacleDetector:
    """HSV detector for simulator obstacles behind the future detector seam."""

    def __init__(self, min_area_ratio: float = 0.001):
        if (isinstance(min_area_ratio, bool)
                or not isinstance(min_area_ratio, (int, float))
                or not math.isfinite(min_area_ratio) or min_area_ratio <= 0):
            raise ValueError('min_area_ratio must be finite and positive')
        self.min_area_ratio = min_area_ratio

    def detect(self, frame: SensorFrame, target_class: str) -> ObjectObservation | None:
        if target_class != 'obstacle' or frame.rgb.size == 0:
            return None
        rgb = np.ascontiguousarray(frame.rgb[..., :3], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (95, 80, 60), (125, 255, 255))
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(contour))
        height, width = rgb.shape[:2]
        area = contour_area / float(height * width)
        if area < self.min_area_ratio:
            return None
        moments = cv2.moments(contour)
        if moments['m00'] <= 0.0:
            return None
        center_x = moments['m10'] / moments['m00']
        bearing = center_x / max(width - 1, 1) * 2.0 - 1.0
        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = (x / width, y / height, (x + box_width) / width, (y + box_height) / height)
        fill = contour_area / float(box_width * box_height)
        confidence = min(1.0, fill * min(1.0, area / self.min_area_ratio))
        return ObjectObservation('obstacle', float(confidence),
                                 tuple(float(value) for value in bbox), float(bearing),
                                 area, frame.timestamp)
