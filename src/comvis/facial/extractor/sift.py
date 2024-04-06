import dataclasses
from typing import final

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from typing_extensions import Self

from comvis.facial.extractor.base import IdentityFeatureExtractor, ExtractedResultLike
from comvis.facial.util import plot_image_sequence

__all__ = ['SIFTExtractorResult',
           'SIFTFeatureExtractor',
           'plot_sift_extracted_result']


@dataclasses.dataclass
class SIFTExtractorResult(ExtractedResultLike):
    image: np.ndarray
    """input image"""
    descriptor: np.ndarray | None
    """SIFT descriptor for the image, 2D array. None if not found"""
    keypoints: np.ndarray
    """extracted keypoints"""

    def flatten(self) -> Self:
        return dataclasses.replace(self, descriptor=self.descriptor.flatten())

    def imshow(self):
        """plot individually"""
        gray = normalize_to_sift(self.image)
        img = cv2.drawKeypoints(gray, self.keypoints, None)
        plt.imshow(img)
        plt.show()


@final
class SIFTFeatureExtractor(IdentityFeatureExtractor):
    """SIFT fracture extraction"""

    def __init__(self, n_features: int = 30):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=self.n_features)

    def __call__(self, X) -> list[SIFTExtractorResult]:
        return self.transform(X)

    def _create_kmeans(self, X, n: int = 50) -> KMeans:
        features = []
        for image in X:
            # ensure image is in grayscale
            if len(image.shape) == 3:
                image = normalize_to_sift(image)

            _, des = self.sift.detectAndCompute(image, None)

            if des is not None:
                features.append(des[:self.n_features, :])

        return kmean_cluster_descriptor(np.vstack(features), n)

    def transform(self, X) -> list[SIFTExtractorResult]:
        """ """
        kmeans = self._create_kmeans(X)

        features = []
        for image in X:
            # ensure image is in grayscale
            if len(image.shape) == 3:
                image = normalize_to_sift(image)

            keypoints, descriptors = self.sift.detectAndCompute(image, None)

            if descriptors is not None:
                histogram = np.zeros(len(kmeans.cluster_centers_))
                clusters = kmeans.predict(descriptors)

                for i in clusters:
                    histogram[i] += 1

                features.append([histogram, keypoints])
            else:
                features.append([np.zeros(len(kmeans.cluster_centers_)), keypoints])

        return [
            SIFTExtractorResult(img, features[i][0], features[i][1])
            for i, img in enumerate(X)
        ]


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


def kmean_cluster_descriptor(descriptors: np.ndarray, n_clusters=50) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)
    return kmeans


def plot_sift_extracted_result(results: list[SIFTExtractorResult]):
    """plot the feature extracted by :class:`SIFTFeatureExtractor`"""
    imgs = []
    for it in results:
        gray = normalize_to_sift(it.image)
        imgs.append(cv2.drawKeypoints(gray, it.keypoints, None))

    imgs = np.array(imgs)
    plot_image_sequence(imgs)
