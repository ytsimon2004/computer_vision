import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.extractor.hog import HOGFeatureExtractor
from comvis.facial.extractor.pca import PCAFeatureExtractor
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.main_feature_repr import FeatureReprOptions

__all__ = ['run_hog_extractor']

from comvis.facial.util import plot_image_sequence


def run_hog_extractor(plot_hog: bool = False):
    """
    :param plot_hog: If True, plot the hog image.
                     If False, evaluate the extract result using tSNE
    :return:
    """
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    hog_extractor = HOGFeatureExtractor()

    if plot_hog:
        ret = hog_extractor(opt.train_X)
        for r in ret:
            r.imshow()
    else:
        hog_extractor.eval_hog_result(opt.train_X, opt.train_y, plot_type='tsne')


def run_pca_extractor(n_components: int = 14):
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()
    opt = FeatureReprOptions(dat, preprocessor)

    pca_extractor = PCAFeatureExtractor(n_components)

    dat = np.delete(opt.train_X, [14, 24, 28, 35, 65], axis=0)
    y = np.delete(opt.train_y, [14, 24, 28, 35, 65], axis=0)

    pca_extractor.fit(dat[y == 2])

    eigenfaces = pca_extractor.get_components().reshape(
        (n_components, opt.image_height, opt.image_width)
    )

    # plot
    plot_image_sequence(eigenfaces, n=n_components, cmap='gray')


if __name__ == '__main__':
    run_pca_extractor()
