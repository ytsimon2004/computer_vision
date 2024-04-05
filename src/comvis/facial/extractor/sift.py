from typing import final, NamedTuple, Literal

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from typing_extensions import Self

from comvis.facial.extractor.base import IdentityFeatureExtractor
from comvis.facial.extractor.plot import plot_as_tsne, plot_as_pca
from comvis.facial.util import plot_image_sequence

__all__ = ['SIFTFeatureExtractor',
           'plot_sift_extracted_result']


class SIFTExtractorResult(NamedTuple):
    image: np.ndarray
    """input image"""
    descriptor: np.ndarray
    """SIFT descriptor for the image"""
    keypoints: np.ndarray
    """extracted keypoint"""

    def flatten(self) -> Self:
        return self._replace(descriptor=self.descriptor.flatten())

    def imshow(self):
        """plot individually"""
        gray = normalize_to_sift(self.image)
        img = cv2.drawKeypoints(gray, self.keypoints, None)
        plt.imshow(img)
        plt.show()


@final
class SIFTFeatureExtractor(IdentityFeatureExtractor):

    def __init__(self, n_features: int = 30):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=self.n_features)

    def __call__(self, X) -> list[SIFTExtractorResult]:
        return self.transform(X)

    def transform(self, X) -> list[SIFTExtractorResult]:
        features = []
        for image in X:
            # ensure image is in grayscale
            if len(image.shape) == 3:
                image = normalize_to_sift(image)

            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            features.append((descriptors, keypoints))

        return [SIFTExtractorResult(img, features[i][0], features[i][1]) for i, img in enumerate(X)]

    # FIXME
    def eval_sift_result(self,
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


# ============ #

def normalize_to_sift(img: np.ndarray) -> np.ndarray:
    """normalize and convert to grayscale for sift"""
    img = rgb2gray(img)

    if img.max() > img.min():
        ret = (255 * ((img - img.min()) / (img.max() - img.min()))).astype(np.uint8)
    else:
        # If the image is completely uniform (all pixels have the same value), just return it as is
        # Or, alternatively, set it to a default value, such as all zeros (black) or all 255 (white)
        ret = img.astype(np.uint8)  # Or, np.zeros(img.shape, dtype=np.uint8) for a black image

    return ret


def plot_sift_extracted_result(results: list[SIFTExtractorResult]):
    """plot the feature extracted by :class:`SIFTFeatureExtractor`"""
    imgs = []
    for it in results:
        gray = normalize_to_sift(it.image)
        imgs.append(cv2.drawKeypoints(gray, it.keypoints, None))

    imgs = np.array(imgs)
    plot_image_sequence(imgs)
