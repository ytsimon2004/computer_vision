from typing import TypedDict, TypeAlias, Literal

import cv2
import numpy as np

from comvis.utils.colors import COLOR_MAGENTA

__all__ = [
    'ProcessParameters',
    'DEFAULT_PROC_PARS',
    #
    'as_gray',
    'as_blur',
    'as_bilateral',
    'as_sharpen',
    'sobel_detect',
    'canny_detect',
    'draw_circle_detect',
    'red_enhancement'
]


class GaussianBlurPars(TypedDict):
    ksize: int
    sigma: float


class BilateralPars(TypedDict):
    d: int
    """	Diameter of each pixel neighborhood that is used during filtering. 
    If it is non-positive, it is computed from sigmaSpace."""
    sigma_color: int
    sigma_space: int


class Filter2DPars(TypedDict):
    """Image Sharpen"""
    kernel: np.ndarray


class ColorGrabPars(TypedDict):
    lower_color: np.ndarray
    """(R,G,B)"""
    upper_color: np.ndarray
    """(R,G,B)"""
    to_color: np.ndarray
    """masking to which color"""


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
    gaussian_blur: GaussianBlurPars
    bilateral: BilateralPars
    filter2d: Filter2DPars

    # edge detect
    sobelX: SobelPars
    sobelY: SobelPars
    sobelXY: SobelPars
    canny: CannyPars

    #
    hough_circles: HoughCirclesPars
    color_grab: ColorGrabPars


DEFAULT_PROC_PARS: ProcessParameters = {
    'gaussian_blur': GaussianBlurPars(ksize=5, sigma=60),
    'bilateral': BilateralPars(d=30, sigma_color=75, sigma_space=75),
    'filter2d': Filter2DPars(
        kernel=np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
    ),
    # detect
    'sobelX': SobelPars(ddepth=None, dx=1, dy=0, ksize=3, scale=1, delta=0),
    'sobelY': SobelPars(ddepth=None, dx=0, dy=1, ksize=3, scale=1, delta=0),
    'sobelXY': SobelPars(ddepth=None, dx=1, dy=1, ksize=3, scale=1, delta=0),
    'canny': CannyPars(lower_threshold=30, upper_threshold=150),
    'hough_circles': HoughCirclesPars(method=cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30, minRadius=10,
                                      maxRadius=30),
    'color_grab': ColorGrabPars(lower_color=np.array([35, 0, 0]),
                                upper_color=np.array([100, 60, 60]),
                                to_color=np.array([255, 0, 0]))
}

# ======================= #
# Image Process Functions #
# ======================= #

ImageType: TypeAlias = np.ndarray


def as_gray(img: ImageType) -> ImageType:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def as_blur(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['gaussian_blur']
    ksize = pars['ksize']
    sigma = pars['sigma']
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def as_bilateral(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['bilateral']
    return cv2.bilateralFilter(img, pars['d'], pars['sigma_color'], pars['sigma_space'])


def as_sharpen(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['filter2d']
    kernel = np.array(pars['kernel'])
    return cv2.filter2D(img, -1, kernel)


def red_enhancement(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    """Grab the red object and do the morphological operations to enhance the mask"""
    pars = proc_dict['color_grab']

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lc, uc = np.array(pars['lower_color']), np.array(pars['upper_color'])
    mask = cv2.inRange(img, lc, uc)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)  # remove noise

    overlay = np.zeros_like(img)
    overlay[opening == 255] = pars['to_color']

    ret = cv2.addWeighted(img, 1, overlay, 1, 0)

    return cv2.cvtColor(ret, cv2.COLOR_RGB2BGR)


def sobel_detect(img: ImageType,
                 proc_dict: ProcessParameters,
                 sobel_command: Literal['sobelX', 'sobelY', 'sobelXY']) -> ImageType:
    # noinspection PyTypedDict
    pars = proc_dict[sobel_command[1:]]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Sobel(img, **pars)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def canny_detect(img: ImageType, proc_dict: ProcessParameters) -> ImageType:
    pars = proc_dict['canny']
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, pars['lower_threshold'], pars['upper_threshold'])
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def draw_circle_detect(img: ImageType, proc_dict: ProcessParameters) -> None:
    pars = proc_dict['hough_circles']
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
