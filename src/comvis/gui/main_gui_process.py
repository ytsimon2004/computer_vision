"""
Prototype
===
key:
    space: pause
     -h: command
     stop while dragging mask
     after dragging, give command for processing
        -> sharpen
        -> denoise
        -> down-sampling
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
from comvis.utils.util_proc_dict import ProcessParameters, DEFAULT_PROC_PARS, load_process_parameter, \
    create_default_json

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

        # io
        if opt.output is None:
            self.output_directory = Path(opt.file).parent
        else:
            self.output_directory = Path(opt.output)

        # proc pars
        if opt.json is not None:
            self.pars: ProcessParameters = load_process_parameter(opt.json)
        else:
            json_file = self.output_directory / 'proc_pars.json'
            create_default_json(json_file)
            self.pars = load_process_parameter(json_file)

        self._output_file = self.output_directory / f'{opt.file}_proc.mp4'
        self._output_logger = self.output_directory / f'{opt.file}_proc_logger.txt'

    def start(self, pause_on_start: bool = True):
        super().start(pause_on_start)

    def handle_command(self, command: str):
        super().handle_command(command)

        match command:
            case ':gray':
                self.enqueue_message('COLOR_BGR2GRAY')
            case ':blur':
                self.enqueue_message('GaussianBlur')
            case ':edge':
                self.enqueue_message('Canny')
            case ':sharpen':
                self.enqueue_message('filter2D')
            case ':r':  # rollback to original
                self.enqueue_message('rollback')

            #
            case ':h':
                self.enqueue_message(':gray    :Image to grayscale')
                self.enqueue_message(':blur    :Blur the image')
                self.enqueue_message(':edge    :Canny Edge detection')
                self.enqueue_message(':sharpen :Sharpen the image')
                self.enqueue_message(':r       :Rollback to original(raw) image')

    def proc_image(self, img: np.ndarray, command: str) -> np.ndarray:
        match command:
            case ':gray':
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            case ':blur':
                pars = self.pars['GaussianBlur']
                ksize = pars['ksize']
                sigma = pars['sigma']
                return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            case ':edge':
                pars = self.pars['Canny']
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return cv2.Canny(img, pars['lower_threshold'], pars['upper_threshold'])
            case ':sharpen':
                pars = self.pars['Filter2D']
                kernel = np.array(pars['kernel'])
                return cv2.filter2D(img, -1, kernel)
            case ':r':
                return img

        return img


if __name__ == '__main__':
    parser = Cv2BasicImageProcessor.cli_parser().parse_args()
    Cv2BasicImageProcessor(parser).start()
