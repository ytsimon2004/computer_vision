from typing import Final

import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.haar_preprocessor import HAARPreprocessor

__all__ = ['FeatureReprOptions']


class FeatureReprOptions:

    def __init__(self, dat: FacialDataSet,
                 preprocessor: HAARPreprocessor):
        self.dat: Final[FacialDataSet] = dat
        self.preproc: Final[HAARPreprocessor] = preprocessor

    @property
    def train_X(self) -> np.ndarray:
        return self.preproc(self.dat.train)

    @property
    def train_y(self) -> np.ndarray:
        return np.array(self.dat.train['class'])

    @property
    def test_X(self) -> np.ndarray:
        return self.preproc(self.dat.test)

    @property
    def image_height(self) -> int:
        """assume image dataset has same dim"""
        return self.train_X[0].shape[0]

    @property
    def image_width(self) -> int:
        """assume image dataset has same dim"""
        return self.train_X[0].shape[1]
