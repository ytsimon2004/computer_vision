"""
Prototype
===
key:
    space: pause
     -h: command
     stop while dragging mask
     after dragging, give command for processing
        -> gray scale
        -> sharpen
        -> denoise
        -> down-sampling
        -> edge detection
        -> gaussian_blur
        -> object tracking? TODO or another child class

    -q: rollback


feature

"""
import logging
from pathlib import Path
from typing import TypedDict, final

import numpy as np

from src.comvis.utils.util_json import load_from_json
from src.comvis.utils.util_proc_dict import ProcessParameters, load_process_parameter
from src.comvis.utils.util_typing import PathLike

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()





@final
class Cv2BasicImageProcessor:

    MOUSE_STATE_FREE = 0  # mouse free moving
    MOUSE_STATE_ROI = 1  # mouse dragging, making roi.

    def __init__(self, file: str,
                 json_file: str | None = None,
                 output_directory: str | None = None):
        super().__init__(file)

        self.pars: ProcessParameters = load_process_parameter(json_file)


        # save output mp4 and logger
        self.output_directory = Path(output_directory)
        if self.output_directory is not None:
            self._save_flag = True
            self._output_file = self.output_directory / f'{file}_proc.mp4'
            self._output_logger = self.output_directory / f'{file}_proc_logger.txt'
        else:
            self._save_flag = False
