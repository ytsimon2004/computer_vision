import numpy as np

from comvis.facial.data import FacialDataSet
from comvis.facial.haar_preprocessor import HAARPreprocessor
from comvis.facial.plot import plot_image_sequence


def run_haar_preproc():
    dat = FacialDataSet.load()
    preprocessor = HAARPreprocessor()

    train_X = preprocessor(dat.train)
    train_y = np.array(dat.train['class'])
    test_X = preprocessor(dat.test)

    plot_image_sequence(train_X[train_y == 0], n=20, imgs_per_row=10)


if __name__ == '__main__':
    run_haar_preproc()
