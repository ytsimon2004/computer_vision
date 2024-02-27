import logging
import os
import time
from typing import ClassVar

import cv2
import numpy as np

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


class Cv2Player:
    window_title: ClassVar = 'Player'

    MOUSE_STATE_FREE = 0  # mouse free moving
    MOUSE_STATE_DRAG = 1  # mouse dragging, making roi.

    def __init__(self, file: str):
        if not os.path.exists(file):
            raise FileNotFoundError(f'{file}')

        self.video_file = file

        self.message_fade_time = 5  # duration of the message start fading

        # display
        self.roi: np.ndarray = np.zeros((0, 5), dtype=int)  # [frame, x0, y0, x1, y1]
        self.show_time: bool = True  # show time bar
        self.show_roi: bool = True  # show roi rectangle

        # video properties
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

        self._current_operation_state = self.MOUSE_STATE_FREE  # mouse state
        self._current_mouse_hover_frame: int | None = None  # mouse point frame on time bar.
        self._current_roi_region: list[int] | None = None  # roi when roi making

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
        vc = self._init_video()

        cv2.namedWindow(self.window_title, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.window_title, onMouse=self.handle_mouse_event)

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

    def handle_mouse_event(self, event: int, x: int, y: int, flag: int, data):

        match event:
            case cv2.EVENT_MOUSEMOVE:  # move or drag
                if self._current_operation_state == self.MOUSE_STATE_FREE:
                    x0, y0, _, _ = self._current_roi_region
                    self._current_roi_region = [x0, y0, x, y]
                else:
                    if self.show_time:
                        self._set_mouse_hover_frame(x, y)
                    else:
                        self._current_mouse_hover_frame = None

            case cv2.EVENT_LBUTTONUP:  # mouse left button up
                if self._current_operation_state == self.MOUSE_STATE_FREE:
                    if self._current_mouse_hover_frame is not None:
                        self.current_frame = self._current_mouse_hover_frame
                    else:
                        self.is_playing = not self.is_playing

            case cv2.EVENT_RBUTTONDOWN:  # mouse right button down
                self._current_operation_state = self.MOUSE_STATE_DRAG
                self._current_roi_region = [x, y, x, y]
                self.is_playing = False

            case cv2.EVENT_RBUTTONUP:
                if self._current_operation_state == self.MOUSE_STATE_DRAG:
                    t = self.current_frame
                    n = self.add_roi(*self._current_roi_region)
                    self.enqueue_message(f'add roi[{n}] at ' + self._frame_to_text(t))
                    self._current_roi_region = None
                    self._current_operation_state = self.MOUSE_STATE_FREE
                else:
                    self._current_operation_state = self.MOUSE_STATE_FREE

    def _set_mouse_hover_frame(self, x: int, y: int):
        """
        calculate the frame where the mouse point on.
        Then update _current_mouse_hover_frame.

        :param x:  mouse x
        :param y:  mouse y
        :return:
        """
        w = self.video_width
        h = self.video_height
        t = self.video_total_frames
        s = 130

        if (h - 20 <= y <= h) and (s <= x <= w - s):
            self._current_mouse_hover_frame = int((x - s) / (w - 2 * s) * t)
        else:
            self._current_mouse_hover_frame = None

    def _frame_to_text(self, frame: int):
        """convert frame to time text, which format '{minute:02d}:{second:02d}'"""
        t_sec = frame // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        return f'{t_min:02d}:{t_sec:02d}'

    # ==== #
    # ROIS #
    # ==== #

    def add_roi(self, x0: int,
                x1: int,
                y0: int,
                y1: int,
                t: int | None = None):
        """
        add roi.
        If there is existed a roi at time t, then replace it with new one.

        :param x0:
        :param x1:
        :param y0:
        :param y1:
        :param t: frame number. If None, use current frame number.
        :return:
        """
        if t is None:
            t = self.current_frame

        i = self._index_roi(t)

        if i is None:
            self.roi = np.sort(np.append(self.roi, np.array([(t, x0, y0, x1, y1)]), axis=0), axis=0)
        else:
            self.roi[i] = (t, x0, y0, x1, y1)

        Logger.info(f'add roi [{t},{x0},{y0},{x1},{y1}]')

        return self._index_roi(t)

    def _index_roi(self, frame: int) -> int | None:
        i = np.nonzero(self.roi[:, 0] == frame)[0]
        if len(i) == 0:
            return None
        else:
            return i[0]
