import numpy as np

from comvis.facial.extractor.base import ExtractedResultLike

__all__ = ['concat_descriptor_result']


def concat_descriptor_result(results: list[ExtractedResultLike]) -> np.ndarray:
    """
    Cancat to 2D array for classifier

    :param results:
    :return:
    """
    return np.vstack([it.flatten().descriptor for it in results])
