import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score

from comvis.facial.classifier.base import BaseClassificationModel
from comvis.facial.classifier.svd import SVCClassificationModel
from comvis.facial.data import FacialDataSet
from comvis.facial.extractor.hog import HOGFeatureExtractor
from comvis.facial.extractor.sift import SIFTFeatureExtractor
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.main_feature_repr import FeatureReprOptions
from comvis.utils.verbose import printdf


# ===============================  #
# model evaluation on test dataset #
# ===============================  #

def predict_test_dataset(classifier: BaseClassificationModel) -> np.ndarray:
    """

    :param classifier: Any classificationModel
    :return:
        1D Array with test data classes label (Ts, )
    """
    dat = FacialDataSet.load(to_pandas=False)
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    classifier.fit(opt.X_train, opt.y_train)
    y_pred = classifier(opt.X_test)

    return y_pred


def eval_accuracy_score_test(classifier: BaseClassificationModel) -> float:
    """

    :param classifier: Any classificationModel
    :return:
        accuracy_score
    """
    y_predict = predict_test_dataset(classifier)
    y_test = ...  # TODO seems missing
    return accuracy_score(y_predict, y_test)


def compare_extractors():
    """example for see the predicted y from X_test"""
    extractor = HOGFeatureExtractor()
    model_hog = SVCClassificationModel(extractor)
    ty_hog = predict_test_dataset(model_hog)

    extractor = SIFTFeatureExtractor(n_features=5)
    model_sift = SVCClassificationModel(extractor)
    ty_sift = predict_test_dataset(model_sift)

    ret = pl.DataFrame().with_columns(
        [
            pl.Series(ty_hog).alias('predict_hog'),
            pl.Series(ty_sift).alias('predict_sift')
        ]
    )

    printdf(ret)


# =============================================================== #
# apply pipeline in train data only (not sure if it's meaningful) #
# =============================================================== #


if __name__ == '__main__':
    compare_extractors()
