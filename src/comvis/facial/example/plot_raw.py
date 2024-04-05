from comvis.facial.data import FacialDataSet
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.main_feature_repr import FeatureReprOptions
from comvis.facial.util import plot_image_sequence


def run_haar_preproc():
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()

    opt = FeatureReprOptions(dat, preprocessor)

    plot_image_sequence(opt.X_train[opt.y_train == 2], n=20, imgs_per_row=10)


if __name__ == '__main__':
    run_haar_preproc()
