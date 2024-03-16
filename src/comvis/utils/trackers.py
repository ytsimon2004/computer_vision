from typing import Callable

import cv2

__all__ = ['OPENCV_OBJ_TRACKERS']

OPENCV_OBJ_TRACKERS: dict[str, Callable] = {
    'kcf': cv2.TrackerKCF_create,
    'csrt': cv2.TrackerCSRT_create,
    'mil': cv2.TrackerMIL_create,
}
