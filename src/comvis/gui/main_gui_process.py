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
import argparse
import logging
from pathlib import Path
from typing import final

import cv2
import numpy as np

from comvis.gui.player import Cv2Player
from comvis.utils.util_proc_dict import ProcessParameters, DEFAULT_PROC_PARS, load_process_parameter

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


@final
class Cv2BasicImageProcessor(Cv2Player):

    @classmethod
    def cli_parser(cls) -> argparse.ArgumentParser:
        ap = super().cli_parser()

        ap.add_argument('--json', help='json file for imaging processing func parameter')
        ap.add_argument('-O', '--output', type=Path, default=None,
                        help='output directory', dest='output')
        return ap

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

        # proc pars
        if opt.json is not None:
            self.pars: ProcessParameters = load_process_parameter(opt.json)
        else:
            self.pars = DEFAULT_PROC_PARS

        # io
        if opt.output is None:
            self.output_directory = Path(opt.file).parent
        else:
            self.output_directory = Path(opt.output)

        self._output_file = self.output_directory / f'{opt.file}_proc.mp4'
        self._output_logger = self.output_directory / f'{opt.file}_proc_logger.txt'

    def start(self, pause_on_start: bool = True):
        super().start(pause_on_start)

    def handle_command(self, command: str):
        super().handle_command(command)

        match command:
            case ':gray':
                self.enqueue_message('Image to grayscale')

            case ':r':  # rollback to original
                self.enqueue_message('rollback')

    def proc_image(self, img: np.ndarray, command: str) -> np.ndarray:
        match command:
            case ':gray':
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            case ':r':
                return img

        return img



if __name__ == '__main__':
    parser = Cv2BasicImageProcessor.cli_parser().parse_args()
    Cv2BasicImageProcessor(parser).start()