import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from comvis.utils.util_typing import PathLike

__all__ = [
    'ProcessParameters',
    'DEFAULT_PROC_PARS',
    #
    'create_default_json',
    'load_process_parameter'
]


class GaussianBlurPars(TypedDict):
    ksize: int
    sigma: float


class Filter2DPars(TypedDict):
    """Image Sharpen"""
    kernel: np.ndarray


class SobelPars(TypedDict):
    ddepth: int | None
    """The depth of the output image"""
    dx: int
    dy: int
    ksize: int
    scale: int
    delta: float


class CannyPars(TypedDict):
    """Canny Edge Detection"""
    lower_threshold: float
    upper_threshold: float


class HoughCirclesPars(TypedDict, total=False):
    method: int
    """Define the detection method. Currently this is the only one available in OpenCV"""
    dp: float
    """The inverse ratio of resolution"""
    minDist: float
    """Minimum distance between detected centers, determined by image height (i.e.,rows/16)"""
    param1: float
    """Upper threshold for the internal Canny edge detector"""
    param2: float
    """Threshold for center detection"""
    minRadius: int
    """Minimum radius to be detected. If unknown, put zero as default"""
    maxRadius: int
    """Maximum radius to be detected. If unknown, put zero as default."""


class ProcessParameters(TypedDict, total=False):
    """For storage the image process parameters, which load from a json file"""
    GaussianBlur: GaussianBlurPars
    Filter2D: Filter2DPars

    # edge detect
    SobelX: SobelPars
    SobelY: SobelPars
    SobelXY: SobelPars
    Canny: CannyPars

    #
    HoughCircles: HoughCirclesPars


DEFAULT_PROC_PARS: ProcessParameters = {
    'GaussianBlur': GaussianBlurPars(ksize=5, sigma=60),
    'Canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'Filter2D': Filter2DPars(
        kernel=np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    ),
    #
    'SobelX': SobelPars(ddepth=None, dx=1, dy=0, ksize=3, scale=1, delta=0),
    'SobelY': SobelPars(ddepth=None, dx=0, dy=1, ksize=3, scale=1, delta=0),
    'SobelXY': SobelPars(ddepth=None, dx=1, dy=1, ksize=3, scale=1, delta=0),
    #
    'HoughCircles': HoughCirclesPars(method=cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30, minRadius=10, maxRadius=30)

}


# ======= #
# JSON IO #
# ======= #

class JsonEncodeHandler(json.JSONEncoder):
    """extend from the JSONEncoder class and handle the conversions in a default method"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


def create_default_json(output_path: PathLike) -> None:
    with open(output_path, "w") as outfile:
        json.dump(DEFAULT_PROC_PARS, outfile, sort_keys=True, indent=4, cls=JsonEncodeHandler)


def load_process_parameter(f: PathLike) -> ProcessParameters:
    with open(f, "r") as file:
        return json.load(file)
