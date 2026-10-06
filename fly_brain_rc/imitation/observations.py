"""Versioned, exclusively observable student inputs. No map/goal coordinates."""
import numpy as np

SCHEMA = 'camera-cv-ranges-ego-v1'
INPUT_SIZE = 19


def encode_observation(sensor, target, obstacle):
    def detected(obj):
        return (1.0,obj.bearing,min(1.0,obj.apparent_area*5)) if obj else (0.0,0.0,0.0)
    ranges = (sensor.tof,sensor.range_left,sensor.range_right,
              sensor.range_side_left,sensor.range_side_right,sensor.range_rear)
    return np.array((*detected(target),*detected(obstacle),
                     *(min(1.2,max(0.0,r))/1.2 for r in ranges),
                     sensor.speed/.35,sensor.yaw_rate/2,
                     sensor.contact_front,sensor.contact_rear,
                     sensor.contact_left,sensor.contact_right,sensor.contact),dtype=np.float32)
