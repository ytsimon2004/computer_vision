import argparse
import logging
from typing import final, ClassVar

import cv2

from comvis.gui.player_GUI import CV2Player
from comvis.utils.trackers import OPENCV_OBJ_TRACKERS
from comvis.utils.colors import COLOR_MAGENTA, COLOR_RED

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


@final
class ObjTrackerPlayer(CV2Player):
    """Track a selected ROIs"""
    window_title: ClassVar[str] = 'Object tracking Player'

    @classmethod
    def cli_parser(cls) -> argparse.ArgumentParser:
        ap = super().cli_parser()
        ap.add_argument('-T', '--tracker',
                        default='csrt',
                        choices=list(OPENCV_OBJ_TRACKERS.keys()),
                        help='which opencv tracker type')

        return ap

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

        self.tracker = OPENCV_OBJ_TRACKERS[opt.tracker]()
        self.enable_roi_selection = False  # use cv2 builtin instead

    def _init_video(self) -> cv2.VideoCapture:
        vc = super()._init_video()

        ret, image = self.video_capture.read()
        cv2.putText(image, 'Select ROI using mouse left button, and press SPACE button', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
        bbox = cv2.selectROI(image, fromCenter=False, showCrosshair=True, printNotice=False)

        try:
            self.tracker.init(image, bbox)
        except cv2.error as e:
            print('Select ROI using mouse left button FIRST!')
            raise RuntimeError(repr(e))

        cv2.destroyWindow('ROI selector')

        return vc

    def _capture_current_image(self, vc: cv2.VideoCapture) -> None:
        if self._is_playing or self.current_image is None:
            ret, image = vc.read()

            if not ret:
                self._is_playing = False
                return

            success, box = self.tracker.update(image)
            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(image, p1, p2, COLOR_MAGENTA, 2, 1)
            else:
                msg = "Tracking failure detected"
                cv2.putText(image, msg, (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                Logger.warning(msg)

            self.current_image = image


if __name__ == '__main__':
    parser = ObjTrackerPlayer.cli_parser().parse_args()
    ObjTrackerPlayer(parser).start()
