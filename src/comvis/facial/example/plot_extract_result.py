import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.extractor.hog import HOGFeatureExtractor, plot_hog_extracted_result
from comvis.facial.extractor.pca import PCAFeatureExtractor
from comvis.facial.extractor.sift import SIFTFeatureExtractor, plot_sift_extracted_result
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.main_feature_repr import FeatureReprOptions
from comvis.facial.util import plot_image_sequence

__all__ = ['run_hog_extractor']


def run_hog_extractor(plot_individual: bool = False,
                      plot_extracted_cls: bool = False):
    """
    :param plot_individual
    :param plot_extracted_cls
    :return:
    """
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    hog_extractor = HOGFeatureExtractor()

    if plot_extracted_cls:
        hog_extractor.eval_hog_result(opt.X_train, opt.y_train, plot_type='tsne')

    if plot_individual:
        ret = hog_extractor(opt.X_train)
        for r in ret:
            r.imshow()
    else:
        ret = hog_extractor(opt.X_train)
        plot_hog_extracted_result(ret)


def run_sift_extractor():
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    sift_extractor = SIFTFeatureExtractor()

    ret = sift_extractor(opt.X_train)

    plot_sift_extracted_result(ret)



def run_pca_extractor(n_components: int = 14):
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    pca_extractor = PCAFeatureExtractor(n_components)

    dat = np.delete(opt.X_train, [14, 24, 28, 35, 65], axis=0)
    y = np.delete(opt.y_train, [14, 24, 28, 35, 65], axis=0)

    pca_extractor.fit(dat[y == 2])

    eigenfaces = pca_extractor.get_components().reshape(
        (n_components, opt.image_height, opt.image_width)
    )

    # plot
    plot_image_sequence(eigenfaces, n=n_components, cmap='gray')


if __name__ == '__main__':
    run_hog_extractor()
