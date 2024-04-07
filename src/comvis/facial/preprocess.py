from typing import Final

import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.haar_preprocessor import HAARPreprocessor

__all__ = ['PreprocTrainTest']


class PreprocTrainTest:
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
    def y_test(self) -> np.ndarray:
        """TODO no test data label (Ts,)"""
        raise NotImplementedError('')

    # ============== #
    # As Actual Name #
    # ============== #

    @property
    def X_train_mila(self) -> np.ndarray:
        return self.X_train[self.y_train == 2]

    @property
    def X_train_jesse(self) -> np.ndarray:
        return self.X_train[self.y_train == 1]

    @property
    def X_train_michael(self) -> np.ndarray:
        name = self.dat.train['name']
        return self.X_train[name == 'Michael_Cera']

    @property
    def X_train_sarah(self) -> np.ndarray:
        name = self.dat.train['name']
        return self.X_train[name == 'Sarah_Hyland']

    # ===== #
    # image #
    # ===== #

    @property
    def image_height(self) -> int:
        """assume image dataset has same dim"""
        return self.X_train[0].shape[0]

    @property
    def image_width(self) -> int:
        """assume image dataset has same dim"""
        return self.X_train[0].shape[1]
