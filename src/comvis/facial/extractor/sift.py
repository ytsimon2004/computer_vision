from typing import final

import cv2
import numpy as np
from skimage.color import rgb2gray

from comvis.facial.extractor.base import IdentityFeatureExtractor

__all__ = ['SIFTFeatureExtractor']


@final
class SIFTFeatureExtractor(IdentityFeatureExtractor):

    def __init__(self, n_features: int = 500):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=self.n_features)

    def __call__(self, X):
        return self.transform(X)

    def transform(self, X):
        features = []
        for image in X:
            # Ensure image is in grayscale
            if len(image.shape) == 3:
                image = rgb2gray(image)

                # normalize and convert for sift
                if image.dtype != np.uint8:
                    image = (255 * ((image - image.min()) / (image.max() - image.min()))).astype(np.uint8)

            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            features.append((keypoints, descriptors))
        return features
