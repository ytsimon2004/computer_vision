from typing import Final

import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.haar_preprocessor import HAARPreprocessor

__all__ = ['FeatureReprOptions']


class FeatureReprOptions:
    """Preprocess (HAARPreprocessor), and get train/test facial dataset

    `Dimension parameters`:

        Tr = number of train dataset
        Ts = number of test dataset

    """

    def __init__(self, dat: FacialDataSet,
                 preprocessor: HAARPreprocessor):
        """

        :param dat: `FacialDataSet`
        :param preprocessor: `HAARPreprocessor`
        """
        self.dat: Final[FacialDataSet] = dat
        self.preproc: Final[HAARPreprocessor] = preprocessor

    @property
    def X_train(self) -> np.ndarray:
        """train image dataset (Tr, )"""
        return self.preproc(self.dat.train)

    @property
    def y_train(self) -> np.ndarray:
        """train dataset label (Tr, )"""
        return np.array(self.dat.train['class'])

    @property
    def X_test(self) -> np.ndarray:
        """test image dataset (Ts, )"""
        return self.preproc(self.dat.test)

    @property
    def y_test(self):
        """TODO no test data label (Ts,)"""
        return

    @property
    def image_height(self) -> int:
        """assume image dataset has same dim"""
        return self.X_train[0].shape[0]

    @property
    def image_width(self) -> int:
        """assume image dataset has same dim"""
        return self.X_train[0].shape[1]
