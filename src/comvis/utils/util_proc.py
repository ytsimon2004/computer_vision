from typing import TypedDict, TypeAlias, Literal

import cv2
import numpy as np

__all__ = [
    'ProcessParameters',
    'DEFAULT_PROC_PARS',
    #
]

from comvis.utils.util_color import COLOR_MAGENTA


class GaussianBlurPars(TypedDict):
    ksize: int
    sigma: float


class Filter2DPars(TypedDict):
    """Image Sharpen"""
    kernel: np.ndarray


class SobelPars(TypedDict):
    ddepth: int | None
    """The depth of the output image"""
    dx: int
    dy: int
    ksize: int
    scale: int
    delta: float


class CannyPars(TypedDict):
    """Canny Edge Detection"""
    lower_threshold: float
    upper_threshold: float


class HoughCirclesPars(TypedDict, total=False):
    method: int
    """Define the detection method. Currently this is the only one available in OpenCV"""
    dp: float
    """The inverse ratio of resolution"""
    minDist: float
    """Minimum distance between detected centers, determined by image height (i.e.,rows/16)"""
    param1: float
    """Upper threshold for the internal Canny edge detector"""
    param2: float
    """Threshold for center detection"""
    minRadius: int
    """Minimum radius to be detected. If unknown, put zero as default"""
    maxRadius: int
    """Maximum radius to be detected. If unknown, put zero as default."""


class ProcessParameters(TypedDict, total=False):
    """For storage the image process parameters, which load from a json file"""
    GaussianBlur: GaussianBlurPars
    Filter2D: Filter2DPars

    # edge detect
    SobelX: SobelPars
    SobelY: SobelPars
    SobelXY: SobelPars
    Canny: CannyPars

    #
    HoughCircles: HoughCirclesPars


DEFAULT_PROC_PARS: ProcessParameters = {
    'GaussianBlur': GaussianBlurPars(ksize=5, sigma=60),
    'Canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'Filter2D': Filter2DPars(
        kernel=np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    ),
    #
    'SobelX': SobelPars(ddepth=None, dx=1, dy=0, ksize=3, scale=1, delta=0),
    'SobelY': SobelPars(ddepth=None, dx=0, dy=1, ksize=3, scale=1, delta=0),
    'SobelXY': SobelPars(ddepth=None, dx=1, dy=1, ksize=3, scale=1, delta=0),
    #
    'HoughCircles': HoughCirclesPars(method=cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30, minRadius=10, maxRadius=30)
}

# ======================= #
# Image Process Functions #
# ======================= #

ImageType: TypeAlias = np.ndarray


def as_gray(img: ImageType) -> ImageType:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def as_blur(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['GaussianBlur']
    ksize = pars['ksize']
    sigma = pars['sigma']
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def as_sharpen(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['Filter2D']
    kernel = np.array(pars['kernel'])
    return cv2.filter2D(img, -1, kernel)


def sobel_detect(img: ImageType,
                 proc_dict: ProcessParameters,
                 sobel_command: Literal['sobelX', 'sobelY', 'sobelXY']) -> ImageType:
    k = sobel_command[1:].replace(sobel_command[1], sobel_command[1].capitalize())
    # noinspection PyTypedDict
    pars = proc_dict[k]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Sobel(img, **pars)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def canny_detect(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['Canny']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, pars['lower_threshold'], pars['upper_threshold'])
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_circle_detect(img: ImageType, proc_dict: ProcessParameters) -> None:
    pars = proc_dict['HoughCircles']
    proc_img = img.copy()
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    proc_img = cv2.medianBlur(proc_img, 5)
    circles = cv2.HoughCircles(proc_img, minDist=img.shape[0] / 40, **pars)

    if circles is not None:
        _draw_circle(img, circles)


def _draw_circle(src: np.ndarray,
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
