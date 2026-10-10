"""Versioned, exclusively observable student inputs. No map/goal coordinates."""
import numpy as np
import cv2

SCHEMA = 'camera-cv-ranges-ego-v1'
INPUT_SIZE = 19
SECTOR_SCHEMA = 'camera-cv-sectors-ranges-ego-v2'
SCHEMA_SIZES = {SCHEMA: INPUT_SIZE, SECTOR_SCHEMA: INPUT_SIZE+12}


def encode_observation(sensor, target, obstacle, *, schema=SCHEMA):
    if schema not in SCHEMA_SIZES:
        raise ValueError('unknown observation schema')
    def detected(obj):
        return (1.0,obj.bearing,min(1.0,obj.apparent_area*5)) if obj else (0.0,0.0,0.0)
    ranges = (sensor.tof,sensor.range_left,sensor.range_right,
              sensor.range_side_left,sensor.range_side_right,sensor.range_rear)
    original = np.array((*detected(target),*detected(obstacle),
                     *(min(1.2,max(0.0,r))/1.2 for r in ranges),
                     sensor.speed/.35,sensor.yaw_rate/2,
                     sensor.contact_front,sensor.contact_rear,
                     sensor.contact_left,sensor.contact_right,sensor.contact),dtype=np.float32)
    if schema == SCHEMA:
        return original
    sectors = np.zeros(12,dtype=np.float32)
    if sensor.rgb.size:
        rgb = np.ascontiguousarray(sensor.rgb[...,:3],dtype=np.uint8)
        hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv,(80,80,60),(125,255,255))
        kernel = np.ones((3,3),dtype=np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        for index,part in enumerate(np.array_split(mask,12,axis=1)):
            if part.size:
                sectors[index] = np.count_nonzero(part)/part.size
    return np.concatenate((original,sectors))
