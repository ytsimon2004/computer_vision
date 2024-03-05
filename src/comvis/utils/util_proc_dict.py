import json
from typing import TypedDict

import numpy as np

from comvis.utils.util_json import load_from_json, JsonEncodeHandler
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

def create_default_json(output_path: PathLike) -> None:
    with open(output_path, "w") as outfile:
        json.dump(DEFAULT_PROC_PARS, outfile, sort_keys=True, indent=4, cls=JsonEncodeHandler)


def load_process_parameter(f: PathLike) -> 'ProcessParameters':
    return load_from_json(f)
