from sklearn.metrics import accuracy_score

from comvis.facial.classifier.base import BaseClassificationModel
from comvis.facial.classifier.svd import SVCClassificationModel
from comvis.facial.data import FacialDataSet
from comvis.facial.extractor.hog import HOGFeatureExtractor
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.main_feature_repr import FeatureReprOptions


def eval_accuracy_score(classifier: BaseClassificationModel):
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    X_train = opt.X_train
    X_test = opt.X_test
    y_train = opt.y_train
    y_test = ...

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(y_pred)
    accuracy_score(y_test, y_pred)


def main():
    extractor = HOGFeatureExtractor()
    model = SVCClassificationModel(extractor)

    eval_accuracy_score(model)


if __name__ == '__main__':
    main()
