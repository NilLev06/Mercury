import numpy as np


class CameraSensor:
    def read(self) -> np.ndarray:
        raise NotImplementedError


class RangeSensor:
    def read(self) -> dict:
        raise NotImplementedError


class Odometry:
    def read(self) -> dict:
        raise NotImplementedError