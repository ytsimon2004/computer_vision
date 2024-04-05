import abc
import dataclasses

import numpy as np
from typing_extensions import Self

__all__ = ['ExtractedResultLike',
           'IdentityFeatureExtractor',
           'concat_descriptor_result']


@dataclasses.dataclass
class ExtractedResultLike(metaclass=abc.ABCMeta):
    descriptor: np.ndarray

    @abc.abstractmethod
    def flatten(self) -> Self:
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
