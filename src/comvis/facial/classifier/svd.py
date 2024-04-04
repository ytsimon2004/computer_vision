from sklearn.svm import SVC

from comvis.facial.classifier.base import BaseClassificationModel
from comvis.facial.extractor.base import IdentityFeatureExtractor

__all__ = ['SVCClassificationModel']


class SVCClassificationModel(BaseClassificationModel):

    def __init__(self, extractor: IdentityFeatureExtractor,
                 kernel='linear'):
        super().__init__(extractor)
        self.classifier = SVC(kernel=kernel, C=1.0)

    def fit(self, X, y):
        features = self.extractor.transform(X)
        self.classifier.fit(features, y)

    def predict(self, X):
        features = self.extractor.transform(X)
        return self.classifier.predict(features)
