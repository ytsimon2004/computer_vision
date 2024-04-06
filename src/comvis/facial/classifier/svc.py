import numpy as np
from sklearn.svm import SVC

from comvis.facial.classifier.base import BaseClassificationModel
from comvis.facial.extractor.base import IdentityFeatureExtractor
from comvis.facial.extractor.util import concat_descriptor_result

__all__ = ['SVCClassificationModel']


class SVCClassificationModel(BaseClassificationModel):
    """SVC classification on the basis extractor"""

    def __init__(self,
                 extractor: IdentityFeatureExtractor,
                 kernel='linear'):
        super().__init__(extractor)
        self.classifier = SVC(kernel=kernel, C=1.0)

    def fit(self, X, y):
        features = self.extractor.transform(X)
        ignore, features = concat_descriptor_result(features)

        #
        mask = np.full_like(y, 1, dtype=bool)
        mask[ignore] = False

        self.classifier.fit(features, y[mask])

    def predict(self, X):
        features = self.extractor.transform(X)
        ignore, features = concat_descriptor_result(features)
        print(f'{features.shape=}')
        return self.classifier.predict(features)
