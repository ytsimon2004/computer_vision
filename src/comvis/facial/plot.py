from matplotlib import pyplot as plt

__all__ = ['plot_image_sequence']


def plot_image_sequence(data,
                        n: int,
                        imgs_per_row: int = 7):
    n_rows = 1 + int(n / (imgs_per_row + 1))
    n_cols = min(imgs_per_row, n)

    f, ax = plt.subplots(n_rows, n_cols)
    for i in range(n):
        if n == 1:
            ax.imshow(data[i])
        elif n_rows > 1:
            ax[int(i / imgs_per_row), int(i % imgs_per_row)].imshow(data[i])
        else:
            ax[int(i % n)].imshow(data[i])
    plt.show()
