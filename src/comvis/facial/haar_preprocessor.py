from pathlib import Path
from typing import ClassVar, Final, Sequence
from urllib import request

import cv2
import numpy as np
import pandas as pd
import polars as pl

from comvis.facial.util import DEFAULT_CACHE_DIRECTORY
from comvis.utils.types import PathLike, DataFrame


class HAARPreprocessor:
    """Preprocessing pipeline built around HAAR feature based cascade classifiers. """
    CASCADES_NAME: ClassVar[str] = 'haarcascade_frontalface_default.xml'

    def __init__(self, directory: PathLike | None = None,
                 face_size: tuple[float, float] = (100, 100)):
        self.face_size = face_size

        self.directory = DEFAULT_CACHE_DIRECTORY if directory is None else Path(directory)
        self.file = self.directory / self.CASCADES_NAME
        if not self.file.exists():
            self._download_model()

        self.classifier: Final[cv2.CascadeClassifier] = cv2.CascadeClassifier(str(self.file))

    def __call__(self, data: DataFrame) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            dat = [self.preprocess(row) for _, row in data.iterrows()]
        elif isinstance(data, pl.DataFrame):
            dat = [self.preprocess(row) for row in data.iter_rows(named=True)]
        else:
            raise TypeError('')

        return np.stack(dat).astype(int)

    def _download_model(self):
        url = f'https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/{self.CASCADES_NAME}'
        with request.urlopen(url) as r, open(self.directory, 'wb') as f:
            f.write(r.read())

    def detect_faces(self, img: np.ndarray) -> Sequence:
        """Detect all faces in an image."""

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.classifier.detectMultiScale(
            img_gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    def extract_faces(self, img: np.ndarray) -> list[np.ndarray]:
        """Returns all faces (cropped) in an image."""

        faces = self.detect_faces(img)

        return [img[y:y + h, x:x + w] for (x, y, w, h) in faces]

    def preprocess(self, data_row):
        faces = self.extract_faces(data_row['img'])

        # if no faces were found, return None
        if len(faces) == 0:
            nan_img = np.empty(self.face_size + (3,))
            nan_img[:] = np.nan
            return nan_img

        # only return the first face
        return cv2.resize(faces[0], self.face_size, interpolation=cv2.INTER_AREA)
