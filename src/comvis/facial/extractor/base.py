import abc

import numpy as np
from typing_extensions import Self

__all__ = ['ExtractedResultLike',
           'IdentityFeatureExtractor',
           'concat_descriptor_result']


class ExtractedResultLike(metaclass=abc.ABCMeta):
    """ABC extractor transformed result"""
    descriptor: np.ndarray

    @abc.abstractmethod
    def flatten(self) -> Self:
        """flatten the descriptor for classifier usage"""
        pass


class IdentityFeatureExtractor(metaclass=abc.ABCMeta):
    """ABC feature extractor"""

    @abc.abstractmethod
    def __call__(self, X) -> list[ExtractedResultLike]:
        return self.transform(X)

    @abc.abstractmethod
    def transform(self, X) -> list[ExtractedResultLike]:
        pass


def concat_descriptor_result(results: list[ExtractedResultLike]):
    return np.array([it.flatten().descriptor for it in results])
