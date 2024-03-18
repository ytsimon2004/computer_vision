import json
from pathlib import Path

import numpy as np

from comvis.utils.process_pars import DEFAULT_PROC_PARS, ProcessParameters
from comvis.utils.types import PathLike


__all__ = [
    'create_default_json',
    'load_process_parameter'
]


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
    if Path(f).suffix != '.json':
        raise ValueError('should be a json file')

    with open(f, "r") as file:
        ret = json.load(file)

    default_keys = list(DEFAULT_PROC_PARS.keys())
    if set(ret.keys()) != set(default_keys):
        raise KeyError(f'file is not complete, should contain all the key {default_keys}')

    return ret
