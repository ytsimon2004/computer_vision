import argparse
import logging
import os
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from comvis.gui.keymap import get_keymapping, KeyMapping, find_key_from_value
from comvis.utils.colors import COLOR_RED, COLOR_YELLOW, COLOR_GREEN, COLOR_CYAN

logging.basicConfig(
    level=logging.DEBUG
)

Logger = logging.getLogger()


class CV2Player:
    window_title: ClassVar[str] = 'Player'

    MOUSE_STATE_FREE = 0  # mouse free moving
    MOUSE_STATE_DRAG = 1  # mouse dragging, making roi.

    @classmethod
    def cli_parser(cls) -> argparse.ArgumentParser:

        ap = argparse.ArgumentParser()
        ap.add_argument('-F', '--file', metavar='FILE', required=True, help='video file', dest='file')
        ap.add_argument('-O', '--output', type=Path, default=None,
                        help='output directory', dest='output')

        return ap

    def __init__(self, opt: argparse.Namespace):

        file = opt.file

        if not os.path.exists(file):
            raise FileNotFoundError(f'{file}')

        self.video_file = file

        # display
        self.roi: np.ndarray = np.zeros((0, 5), dtype=int)  # [frame, x0, y0, x1, y1]
        self.show_time: bool = True  # show time bar
        self.show_roi: bool = True  # show roi rectangle
        self.mouse_stick_to_roi = True  # stick mouse to roi frame on time bar
        self.mouse_stick_distance = 5  # diameter of region to stick mouse on roi frame on time bar

        # video properties
        self.video_capture: cv2.VideoCapture | None = None
        self.current_image: np.ndarray | None = None
        self.video_width: int = 0
        self.video_height: int = 0
        self.video_fps: int = 1
        self.video_total_frames: int = 0

        # control
        self._speed_factor: float = 1  # playing speed factor
        self._sleep_interval = 1  # playing FPS control
        self._is_playing = False  # play/pause
        self.enable_roi_selection = True  # enable customized roi selection
        self._current_operation_state = self.MOUSE_STATE_FREE  # mouse state
        self._current_mouse_hover_frame: int | None = None  # mouse point frame on time bar.
        self._current_roi_region: list[int] | None = None  # roi when roi making
        self._message_queue: list[tuple[float, str]] = []
        self._message_fade_time = 10  # duration of the message start fading
        self.buffer = ''  # input buffer

        # children used
        self._proc_image_command: str = ''

        # io
        self.output_file: Path | None = opt.output

    @property
    def speed_factor(self) -> float:
        """playing speed factor"""
        return self._speed_factor

    @speed_factor.setter
    def speed_factor(self, value: float):
        value = min(32, max(0.25, value))
        self._speed_factor = value
        self._sleep_interval = 1 / self.video_fps / value
        self.enqueue_message(f'speed x{value}')

    @property
    def current_frame(self) -> int:
        """
        :return: current frame number
        :raise: RuntimeError: video file doesn't open yet
        """
        vc = self.video_capture
        if vc is None:
            raise RuntimeError('')

        return int(vc.get(cv2.CAP_PROP_POS_FRAMES))

    @current_frame.setter
    def current_frame(self, value: int):
        """
        set/jump to frame

        :param value: frame number
        :return:
        """
        if not (0 <= value <= self.video_total_frames):
            Logger.warning(f'{value} out of range')
            return

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
        cv2.setMouseCallback(self.window_title, self.handle_mouse_event)

        #
        if self.output_file is not None:
            Logger.info(f'save output in {str(self.output_file)}')

            if self.output_file.suffix == '.mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.output_file.suffix == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            else:
                raise RuntimeError(f'invalid output suffix: {self.output_file.suffix}, only support .mp4 and .avi')

            output = cv2.VideoWriter(str(self.output_file), fourcc, 30.0, (self.video_width, self.video_height))
        else:
            output = None

        #
        try:
            self._is_playing = not pause_on_start
            self._loop(output)
        except KeyboardInterrupt:
            pass
        finally:
            Logger.debug('closing')
            vc.release()
            cv2.destroyWindow(self.window_title)
            Logger.debug('closed')

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

    def _loop(self, output: cv2.VideoWriter | None = None):
        """frame update look."""
        while True:
            t = time.time()
            try:
                self._update(output)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                Logger.warning(e, exc_info=True)
                raise

            t = self._sleep_interval - (time.time() - t)
            if t > 0:
                time.sleep(t)

    def _update(self, output: cv2.VideoWriter | None = None):
        """frame update TODO add realtime fps info"""
        vc = self.video_capture
        self._capture_current_image(vc)

        # copy image for UI drawing.
        image = self.current_image.copy()
        roi = self.current_roi

        # show input buffer content
        if len(self.buffer):
            self._show_buffer(image)

        # display enqueued message
        self._show_queued_message(image)

        # drawing roi
        if self.enable_roi_selection:
            if self._current_roi_region is not None:  # when a roi is creating, draw it first
                self._show_roi_tmp(image)
            elif self.show_roi:  # then current roi.
                if roi is not None:
                    self._show_roi(image, roi)

        # show time bar
        if self.show_time:
            self._show_time_bar(image)

        # update frame
        image = self.proc_image(image, self._proc_image_command)
        cv2.imshow(self.window_title, image)

        # write output
        if output is not None:
            output.write(image)

        # get keyboard input.
        k = cv2.waitKey(1)
        if k > 0:
            self.handle_keycode(k)

    def _capture_current_image(self, vc: cv2.VideoCapture) -> None:
        """get image"""
        if self._is_playing or self.current_image is None:
            ret, image = vc.read()

            if not ret:
                self._is_playing = False
                return

            self.current_image = image

    def proc_image(self, img: np.ndarray, command: str) -> np.ndarray:
        """overwrite by child, image processing"""
        return img

    # ==== #
    # Show #
    # ==== #

    def _show_queued_message(self, image):
        """drawing enqueued message"""
        t = time.time()
        y = 70
        s = 35
        i = 0
        while i < len(self._message_queue):
            r, m = self._message_queue[i]
            if r + self._message_fade_time < t:  # message has showed enough time, so we delete it.
                del self._message_queue[i]
            else:
                cv2.putText(image, m, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CYAN, 2, cv2.LINE_AA)
                i += 1
                y += s

    def _show_buffer(self, image):
        """drawing input buffer content"""
        cv2.putText(image, self.buffer, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

    def _show_roi(self, image, roi: np.ndarray):
        """drawing roi"""
        _, x0, y0, x1, y1 = roi
        cv2.rectangle(image, (x0, y0), (x1, y1), COLOR_YELLOW, 2, cv2.LINE_AA)

    def _show_roi_tmp(self, image):
        """drawing making roi"""
        x0, y0, x1, y1 = self._current_roi_region
        cv2.rectangle(image, (x0, y0), (x1, y1), COLOR_GREEN, 2, cv2.LINE_AA)

    def _show_time_bar(self, image):
        """drawing time bar"""
        s = 130
        w = self.video_width
        h = self.video_height
        frame = self.current_frame

        # total frame text
        cv2.putText(image, self._frame_to_text(self.video_total_frames), (w - 100, h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        # current frame text
        cv2.putText(image, self._frame_to_text(frame), (10, h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2, cv2.LINE_AA)

        # line
        cv2.line(image, (s, h - 10), (w - s, h - 10), COLOR_RED, 3, cv2.LINE_AA)

        #
        # roi frame
        mx = self._frame_to_time_bar_x(self.roi[:, 0])
        for x in mx:
            cv2.line(image, (x, h - 20), (x, h), COLOR_YELLOW, 3, cv2.LINE_AA)

        # current frame
        x = self._frame_to_time_bar_x(frame)
        cv2.line(image, (x, h - 20), (x, h), COLOR_RED, 3, cv2.LINE_AA)

        # mouse hover
        if self._current_mouse_hover_frame is not None:
            x = self._frame_to_time_bar_x(self._current_mouse_hover_frame)

            color = COLOR_GREEN
            if self.mouse_stick_to_roi and len(mx) > 0:
                i = np.argmin(np.abs(mx - x))
                if abs(mx[i] - x) < self.mouse_stick_distance:
                    x = mx[i]
                    self._current_mouse_hover_frame = int(self.roi[i, 0])
                    color = COLOR_YELLOW

            cv2.line(image, (x, h - 20), (x, h), color, 3, cv2.LINE_AA)

            # text
            cv2.putText(image, self._frame_to_text(self._current_mouse_hover_frame), (x - s // 2, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    def _frame_to_time_bar_x(self, frame):
        """
        calculate the position for the frame on time bar.

        Parameters
        ----------
        frame : int or np.ndarray
            frame number.

        Returns
        -------
        x
            x value

        Raises
        ------
        TypeError

        """
        w = self.video_width
        t = self.video_total_frames
        s = 130

        if isinstance(frame, int):
            return int((w - 2 * s) * frame / t) + s
        elif isinstance(frame, np.ndarray):
            return ((w - 2 * s) * frame.astype(float) / t).astype(int) + s
        else:
            raise TypeError(type(frame))

    # ======= #

    def handle_mouse_event(self, event: int, x: int, y: int, flag: int, data):

        match event:
            case cv2.EVENT_MOUSEMOVE:  # move or drag
                if self._current_operation_state == self.MOUSE_STATE_DRAG:
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

    def _frame_to_text(self, frame: int) -> str:
        """convert frame to time text, which format '{minute:02d}:{second:02d}'"""
        t_sec = frame // self.video_fps
        t_min, t_sec = t_sec // 60, t_sec % 60
        return f'{t_min:02d}:{t_sec:02d}'

    # ==== #
    # ROIS #
    # ==== #

    @property
    def current_roi(self) -> np.ndarray | None:
        """current using roi according to current frame"""
        return self.get_roi(self.current_frame)

    def get_roi(self, frame: int) -> np.ndarray | None:
        """get roi according to frame"""
        if not (0 <= frame <= self.video_total_frames):
            raise ValueError('')

        ret = None
        for mask in self.roi:
            t = mask[0]
            if t <= frame:
                ret = mask
        return ret

    def add_roi(self, x0: int,
                y0: int,
                x1: int,
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

    # ============= #
    # Key & Command #
    # ============= #

    def goto_begin(self):
        self.current_frame = 0

    def goto_end(self):
        self.current_frame = self.video_total_frames

    def handle_keycode(self, k: int):
        # Logger.debug(f'Key: {k}')

        mapping = get_keymapping()
        ret = self._handle_keymapping(mapping, k)
        if ret is not None:  # printable
            self.buffer += chr(k)

    def _handle_keymapping(self, mapping: KeyMapping, value: int) -> int | None:
        """
        Handling the keyboard mapping
        :param mapping:
        :param value:
        :return: int value if cannot find key in keymapping, otherwise return None
        """
        ret = find_key_from_value(mapping, value)
        if not ret:
            return value

        match ret:
            case 'space':
                self.is_playing = not self.is_playing
            case 'left':
                self.current_frame -= 10
            case 'right':
                self.current_frame += 10
            case 'left_square_bracket':
                self.goto_begin()
            case 'right_square_bracket':
                self.goto_end()
            case 'plus':
                self.speed_factor *= 2
            case 'minus':
                self.speed_factor /= 2
            case 'backspace':
                if len(self.buffer) > 0:
                    self.buffer = self.buffer[:-1]
            case 'enter':  # handle command in buffer
                command = self._proc_image_command = self.buffer
                self.buffer = ''
                try:
                    self.handle_command(command)
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    self.enqueue_message(f'command "{command}" {type(e).__name__}: {e}')
            case 'escape':
                self.buffer = ''

    def handle_command(self, command: str):
        Logger.debug(f'command: {command}')

        match command:
            case ':h':
                self.enqueue_message('space        :play/pause')
                self.enqueue_message('left/right   :go backward/forward 10 frames')
                self.enqueue_message('[/]          :go backward/forward 10 frames')
                self.enqueue_message('+/-          :adjust speed')
            case ':q':
                raise KeyboardInterrupt
            case ':d':  # delete roi:d
                self.roi = np.zeros((0, 5), dtype=int)


if __name__ == '__main__':
    parser = CV2Player.cli_parser().parse_args()
    CV2Player(parser).start()
