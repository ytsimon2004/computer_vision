"""
Prototype
===
key:
    space: pause
     -h: command
     stop while dragging mask
     after dragging, give command for processing
        -> denoise
        -> down-sampling

    -q: rollback


feature

"""
import argparse
import logging
from pathlib import Path
from typing import final

import cv2
import numpy as np

from comvis.gui.io import create_default_json, load_process_parameter
from comvis.gui.player_GUI import CV2Player
from comvis.utils.util_color import COLOR_MAGENTA
from comvis.utils.util_proc import (
    as_gray,
    as_blur,
    sobel_detect,
    as_sharpen,
    canny_detect,
    draw_circle_detect,
    as_bilateral,
    red_enhancement
)
from comvis.utils.util_type import PathLike

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


@final
class ImageProcPlayer(CV2Player):
    """Video Player for demonstrate the effect with cv2 image processing function"""

    @classmethod
    def cli_parser(cls) -> argparse.ArgumentParser:
        ap = super().cli_parser()

        ap.add_argument('--json', type=Path, help='json file for imaging processing func parameter')
        ap.add_argument('-O', '--output', type=Path, default=None,
                        help='output directory', dest='output')
        return ap

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

        self.output_file: Path | None = opt.output

        # proc pars
        if opt.json is not None:
            self.pars = load_process_parameter(opt.json)
        else:
            json_file = Path(self.video_file).parent / 'proc_pars.json'
            create_default_json(json_file)
            self.pars = load_process_parameter(json_file)

    def start(self, pause_on_start: bool = True, output: PathLike | None = None):
        super().start(pause_on_start, self.output_file)

    def handle_command(self, command: str):
        super().handle_command(command)

        match command:
            # TODO update in more comprehensive
            case ':gray':
                self.enqueue_message('COLOR_BGR2GRAY')
            case ':blur':
                self.enqueue_message('GaussianBlur')
            case ':sharpen':
                self.enqueue_message('filter2D')
            case ':sobelX' | ':sobelY' | ':sobelXY':
                self.enqueue_message('Sobel')
            case ':canny':
                self.enqueue_message('Canny')
            case ':red':
                self.enqueue_message('>> grab red object and enhance')
            case 'bilateral':
                self.enqueue_message('>> bilateral filter smoothing')
            case ':r':  # rollback to original
                self.enqueue_message('rollback')

            #
            case ':h':
                self.enqueue_message(':d            :Delete the ROI')
                self.enqueue_message(':q            :Exit the GUI')
                self.enqueue_message(':gray         :Image to grayscale')
                self.enqueue_message(':blur         :Gaussian blur the image')
                self.enqueue_message(':red          :Grab the red object and enhance the brightness')
                self.enqueue_message(':bilateral    :Bilateral filter the image')
                self.enqueue_message(':sharpen      :Sharpen the image')
                self.enqueue_message(':sobel        :Sobel Edge detection')
                self.enqueue_message(':canny        :Canny Edge detection')
                self.enqueue_message(':circle       :Circular detection')
                self.enqueue_message(':r            :Rollback to original(raw) image')

    def proc_image(self, img: np.ndarray, command: str) -> np.ndarray:
        proc = self._get_proc_part(img)

        match command:
            case ':gray':
                proc = as_gray(proc)
            case ':blur':
                proc = as_blur(proc, self.pars)
            case ':bilateral':
                proc = as_bilateral(proc, self.pars)
            case ':sharpen':
                proc = as_sharpen(proc, self.pars)
            case ':red':
                proc = red_enhancement(proc, self.pars)
            case ':sobelX' | ':sobelY' | ':sobelXY':
                # noinspection PyTypeChecker
                proc = sobel_detect(proc, self.pars, command)
            case ':canny':
                proc = canny_detect(proc, self.pars)
            case ':circle':
                draw_circle_detect(proc, self.pars)
            case ':r':
                return img
            case _:
                return img

        if (roi := self.current_roi) is not None:
            _, x0, y0, x1, y1 = roi
            img[y0:y1, x0:x1] = proc
            return img

        return proc

    def _get_proc_part(self, img: np.ndarray) -> np.ndarray:
        """get the image part need to be processed"""
        if (roi := self.current_roi) is not None:
            _, x0, y0, x1, y1 = roi
            return img[y0:y1, x0:x1]
        else:
            return img

    @staticmethod
    def draw_detected_circle(src: np.ndarray,
                             circles: np.ndarray):
        """
        :param src: source image
        :param circles: A vector that stores sets of 3 values: xc,yc,r for each detected circle
        :return:
        """
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(src, center, 1, (0, 100, 100), 3)  # center
            radius = i[2]
            cv2.circle(src, center, radius, COLOR_MAGENTA, 3)  # circle outline


if __name__ == '__main__':
    parser = ImageProcPlayer.cli_parser().parse_args()
    ImageProcPlayer(parser).start()
