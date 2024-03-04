from typing import TypedDict

from comvis.utils.util_json import load_from_json
from comvis.utils.util_typing import PathLike

__all__ = [
    'ProcessParameters',
    'DEFAULT_PROC_PARS',
    'load_process_parameter'
]


class ProcessParameters(TypedDict, total=False):
    """For storage the image process parameters, which load from a json file"""
    pass


DEFAULT_PROC_PARS: ProcessParameters = {

}


def load_process_parameter(f: PathLike) -> 'ProcessParameters':
    return load_from_json(f)
