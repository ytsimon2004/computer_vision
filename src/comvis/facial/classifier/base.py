import abc

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
    def fit(self, X, y):
        """Fits the classifier to the training data"""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Predicts labels for new data."""
        pass
