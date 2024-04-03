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
                        n: int,
                        imgs_per_row: int = 7,
                        **kwargs):
    """

    :param data:
    :param n:
    :param imgs_per_row:
    :return:
    """
    n_rows = 1 + int(n / (imgs_per_row + 1))
    n_cols = min(imgs_per_row, n)

    f, ax = plt.subplots(n_rows, n_cols)
    for i in range(n):
        if n == 1:
            ax.imshow(data[i], **kwargs)
        elif n_rows > 1:
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].imshow(data[i], **kwargs)
        else:
            ax[int(i % n)].imshow(data[i], **kwargs)
    plt.show()
