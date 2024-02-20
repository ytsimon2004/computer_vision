import logging
import os
import time
from typing import ClassVar

import cv2

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


class Cv2Player:
    window_title: ClassVar = 'Player'

    def __init__(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f'{file}')

        self.video_file = file

        self.message_fade_time = 5  # duration of the message start fading

        #
        self.video_capture: cv2.VideoCapture | None = None
        self.current_image: int | None = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.video_fps: int = 1
        self.video_total_frames: int = 0
        self.current_image = None

        # control
        self._speed_factor: float = 1  # playing speed factor
        self._sleep_interval = 1  # playing FPS control
        self._is_playing = False  # play/pause
        self._message_queue: list[tuple[float, str]] = []

    @property
    def speed_factor(self) -> float:
        """playing speed factor"""
        return self._speed_factor

    @speed_factor.setter
    def speed_factor(self, value: float):
        value = min(32, max(0.25, value))
        self._speed_factor = value
        self._sleep_interval = 1 / self.video_fps / value

    @property
    def current_frame(self) -> int:
        """
        :return: current frame number
        :raise: RuntimeError: video file doesn't open yet
        """
        vc = self.video_capture
        if vc is None:
            raise RuntimeError('')

        return int(vc.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    @current_frame.setter
    def current_frame(self, value: int):
        """
        set/jump to frame

        :param value: frame number
        :return:
        """
        if not (0 <= value < self.video_total_frames):
            raise ValueError()

        if (vc := self.video_capture) is not None:
            vc.set(cv2.CAP_PROP_POS_FRAMES, value - 1)

        self.current_image = None

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @is_playing.setter
    def is_playing(self, value: bool):
        """set playing or pause."""
        self._is_playing = value
        if value:
            Logger.debug('play')
        else:
            Logger.debug('pause')

    def enqueue_message(self, text: str):
        """enqueue message to queue"""
        self._message_queue.append((time.time(), text))

    # ====== #

    def start(self, pause_on_start: bool = True):
        Logger.debug('start the GUI')
        # TODO fromhere

    def _init_video(self) -> cv2.VideoCapture:
        Logger.debug(f'file = {self.video_file}')

        self.video_capture = vc = cv2.VideoCapture(self.video_file)
        self.video_width = w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Logger.debug(f'width,height = {w},{h}')

        self.video_fps = fps = int(vc.get(cv2.CAP_PROP_FPS))
        Logger.debug(f'fps = {fps}')

        self.speed_factor = self._speed_factor  # update sleep_interval
        self.video_total_frames = f = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        Logger.debug(f'total_frame = {f}')

        return vc
