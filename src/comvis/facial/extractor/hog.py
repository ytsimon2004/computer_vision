import dataclasses
from typing import Literal, final

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.exposure import exposure
from skimage.feature import hog
from typing_extensions import Self

from comvis.facial.extractor.base import IdentityFeatureExtractor, ExtractedResultLike
from comvis.facial.extractor.plot import plot_as_tsne, plot_as_pca
from comvis.facial.util import plot_image_sequence

__all__ = ['HogExtractorResult',
           'HOGFeatureExtractor',
           'plot_hog_extracted_result']

HOG_BLOCK_NORM_METHOD = Literal['L1’', 'L1-sqrt', 'L2', 'L2-Hys']


@dataclasses.dataclass
class HogExtractorResult(ExtractedResultLike):
    """Container for storing the hog results"""
    image: np.ndarray
    """input image"""
    descriptor: np.ndarray
    """HOG descriptor for the image"""
    hog_image: np.ndarray
    """A visualisation of the HOG image"""

    def flatten(self) -> Self:
        return dataclasses.replace(self, descriptor=self.descriptor.flatten())

    def imshow(self) -> None:
        """see pair by pair"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(self.image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(self.hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()


@final
class HOGFeatureExtractor(IdentityFeatureExtractor):
    """Wrapper for HOG in skimage.feature"""

    def __init__(self,
                 orientations: int = 8,
                 pixels_per_cell: tuple[int, int] = (16, 16),
                 cells_per_block: tuple[int, int] = (1, 1),
                 block_norm: HOG_BLOCK_NORM_METHOD = 'L2-Hys',
                 **kwargs):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.cell_per_block = block_norm
        self.block_norm = block_norm

        self._kwarg = kwargs

    def __call__(self, X: np.ndarray) -> list[HogExtractorResult]:
        """

        :param X: batch images
        :return:
        """
        return self.transform(X)

    def transform(self, X: np.ndarray) -> list[HogExtractorResult]:
        return [self._hog(img) for img in X]

    def _hog(self, img: np.ndarray) -> HogExtractorResult:
        """
        :param img: image array
        :return:
            HogResult
        """
        if not isinstance(img, np.ndarray):
            raise TypeError('')

        # to grayscale
        _img = rgb2gray(img) if img.ndim == 3 else img

        ret = hog(
            _img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True,
            visualize=True,
            **self._kwarg
        )

        return HogExtractorResult(
            img,
            *ret
        )

    def eval_hog_result(self,
                        X: np.ndarray,
                        labels: np.ndarray,
                        plot_type: Literal['pca', 'tsne'] = 'tsne'):
        assert len(X) == len(labels), 'number not the same between image and label'

        features = self.transform(X)

        match plot_type:
            case 'tsne':
                plot_as_tsne(features, labels)
            case 'pca':
                plot_as_pca(features, labels)


# ================== #

def plot_hog_extracted_result(results: list[HogExtractorResult]):
    hogs = []
    for it in results:
        hog_image_rescaled = exposure.rescale_intensity(it.hog_image, in_range=(0, 10))
        hogs.append(hog_image_rescaled)

    imgs = np.array(hogs)
    plot_image_sequence(imgs, imgs_per_row=7, cmap='gray')
