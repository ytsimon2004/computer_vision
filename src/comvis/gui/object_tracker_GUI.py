import argparse
from typing import final

import cv2
import numpy as np

from comvis.gui.player_GUI import CV2Player
from comvis.utils.trackers import OPENCV_OBJ_TRACKERS


@final
class ObjTrackerPlayer(CV2Player):
    """TODO"""

    @classmethod
    def cli_parser(cls) -> argparse.ArgumentParser:
        ap = super().cli_parser()
        ap.add_argument('-T', '--tracker', default='kcf', choices=list(OPENCV_OBJ_TRACKERS.keys()),
                        help='opencv tracker type')

        return ap

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)
        print(f'{opt.tracker=}')
        self.tracker = OPENCV_OBJ_TRACKERS[opt.tracker]()

    def _update(self, output: cv2.VideoWriter | None = None):
        super()._update(output)

        if (roi := self.current_roi) is not None:
            self.tracker.init(self.current_frame, roi[1:])
            flag, box = self.tracker.update(self.current_frame)

            if flag:
                # Tracking success
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(self.current_frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # Tracking failure
                cv2.putText(self.current_frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def proc_image(self, img: np.ndarray, command: str) -> np.ndarray:
        return img


if __name__ == '__main__':
    parser = ObjTrackerPlayer.cli_parser().parse_args()
    ObjTrackerPlayer(parser).start()
