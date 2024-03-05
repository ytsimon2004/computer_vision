import json
from pathlib import Path
from typing import TypedDict

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


class CannyPars(TypedDict):
    """Canny Edge Detection"""
    lower_threshold: float
    upper_threshold: float


class Filter2DPars(TypedDict):
    """Image Sharpen"""
    kernel: np.ndarray


class ProcessParameters(TypedDict, total=False):
    """For storage the image process parameters, which load from a json file"""
    GaussianBlur: GaussianBlurPars
    Canny: CannyPars
    Filter2D: Filter2DPars


DEFAULT_PROC_PARS: ProcessParameters = {
    'GaussianBlur': GaussianBlurPars(ksize=5, sigma=60),
    'Canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'Filter2D': Filter2DPars(kernel=np.array([[-1, -1, -1],
                                              [-1, 9, -1],
                                              [-1, -1, -1]]))
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
