from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

__all__ = [
    'DEFAULT_CACHE_DIRECTORY',
    #
    'plot_image_sequence'
]

DEFAULT_CACHE_DIRECTORY = Path.home() / '.cache' / 'comvis' / 'facial'


def plot_image_sequence(data: np.ndarray,
                        imgs_per_row: int = 7,
                        **kwargs):
    """
    Plots a sequence of images in a grid

    :param data: Array of images to plot. (N, (*img))
    :param imgs_per_row: Number of images per row in the grid
    :return:
    """
    n = data.shape[0]

    n_rows = np.ceil(n / imgs_per_row).astype(int)
    n_cols = min(imgs_per_row, n)

    _, ax = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows), squeeze=False)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, imgs_per_row)
        ax[r, c].axis('off')  # Hide axes

        if i < n:  # check to avoid index error
            ax[r, c].imshow(data[i], **kwargs)
        else:  # blank out remaining subplots if any
            ax[r, c].set_visible(False)

    plt.show()
