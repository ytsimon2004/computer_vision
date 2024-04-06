import numpy as np

from comvis.facial.extractor.base import ExtractedResultLike
from comvis.facial.extractor.sift import SIFTExtractorResult

__all__ = ['concat_descriptor_result']


def concat_descriptor_result(results: list[ExtractedResultLike]) -> tuple[np.ndarray, np.ndarray]:
    """

    :param results:
    :return:
    """
    if isinstance(results[0], SIFTExtractorResult):
        no_descriptor_idx = []
        desc = []
        for i, it in enumerate(results):
            if it.descriptor is not None:
                desc.append(it.flatten().descriptor)
            else:
                no_descriptor_idx.append(i)

        desc = _handle_descriptor_len_diff(desc)
        print(f'{no_descriptor_idx=}')

    else:
        no_descriptor_idx = []
        desc = [it.flatten().descriptor for it in results if it.descriptor is not None]

    return np.array(no_descriptor_idx).astype(int), np.vstack(desc)


def _handle_descriptor_len_diff(vectors: list[np.ndarray]):
    """

    :param vectors: descriptor with different length. list of 1D array
    :return:
    """
    max_length = max(len(vec) for vec in vectors)
    padded_vectors = [np.pad(vec, (0, max_length - len(vec)), 'constant', constant_values=0) for vec in vectors]

    return np.array(padded_vectors)
