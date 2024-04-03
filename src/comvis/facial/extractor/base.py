import abc

__all__ = ['IdentityFeatureExtractor']


class IdentityFeatureExtractor(metaclass=abc.ABCMeta):
    """ABC feature extractor"""

    @abc.abstractmethod
    def __call__(self, X):
        return self.transform(X)

    @abc.abstractmethod
    def transform(self, X):
        pass
