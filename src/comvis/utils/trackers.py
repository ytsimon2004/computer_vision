from typing import Callable

import cv2

OPENCV_OBJ_TRACKERS: dict[str, Callable] = {
    'kcf': cv2.TrackerKCF_create,
    'csrt': cv2.TrackerCSRT_create
}
