from typing import final

import numpy as np
from sklearn.decomposition import PCA

from comvis.facial.extractor.base import IdentityFeatureExtractor


@final
class PCAFeatureExtractor(IdentityFeatureExtractor):

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess_image(X)
        return self.pca.fit_transform(X)

    def inverse_transform(self, X: np.ndarray):
        X = self._preprocess_image(X)
        return self.pca.inverse_transform(X)

    def fit(self, X: np.ndarray):
        X = self._preprocess_image(X)
        self.pca.fit(X)

    @staticmethod
    def _preprocess_image(X: np.ndarray) -> np.ndarray:
        # to grayscale
        gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

        # flatten
        n_images, height, width = gray.shape
        ret = gray.reshape(n_images, height * width)

        return ret

    def get_components(self):
        return self.pca.components_
