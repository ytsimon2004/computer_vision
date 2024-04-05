import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_as_pca(features: list, labels: np.ndarray) -> None:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pc_feature = pca.fit_transform(np.array([ret.descriptor for ret in features]))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pc_feature[:, 0], pc_feature[:, 1], c=labels, cmap='viridis', alpha=0.6,
                          edgecolors='w')
    plt.title("HOG Features Visualized with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
    plt.show()


def plot_as_tsne(features: list, labels: np.ndarray) -> None:
    tsne = TSNE(n_components=2, random_state=42)

    # desc = []
    # for ret in features:
    #     if ret.descriptor is not None:
    #         _des = ret.descriptor[:30, :]
    #         desc.append(_des)
    #         print(_des.shape)
    #
    # desc = np.vstack(desc)
    # # print(desc.shape)

    tsne_results = tsne.fit_transform(np.array([ret.descriptor for ret in features]))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='w')
    plt.title("t-SNE Visualization of HOG Features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label='Labels')
    plt.show()
