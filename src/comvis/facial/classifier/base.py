import abc

import numpy as np

from comvis.facial.extractor.base import IdentityFeatureExtractor

__all__ = ['BaseClassificationModel']


class BaseClassificationModel(metaclass=abc.ABCMeta):
    classifier = None  # classifier model

    def __init__(self, extractor: IdentityFeatureExtractor):
        """

        :param extractor: An instance of a feature extraction class.
        """
        self.extractor = extractor

    def __call__(self, X):
        return self.predict(X)

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the classifier to the training data

        :param X: X_train (Tr, )
        :param y: y_train (Tr, )
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts labels for new data

        :param X: X_test (Ts,)
        :return: y_test  (Ts,)
        """
        pass
