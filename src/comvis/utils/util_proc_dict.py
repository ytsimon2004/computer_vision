from typing import TypedDict

from src.comvis.utils.util_json import load_from_json
from src.comvis.utils.util_typing import PathLike


class ProcessParameters(TypedDict):
    """For storage the image process parameters, which load from a json file"""
    pass


def load_process_parameter(f: PathLike) -> 'ProcessParameters':
    return load_from_json(f)