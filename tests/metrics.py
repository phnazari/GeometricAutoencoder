import numpy as np
import scipy.sparse
from pykeops.torch import LazyTensor
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr


def corr_pdist_subsample(x, y, sample_size, seed=0, metric="euclidean"):
    """
    Computes correlation between pairwise distances among the x's and among the y's
    :param x: array of positions for x
    :param y: array of positions for y
    :param sample_size: number of points to subsample from x and y for pairwise distance computation
    :param seed: random seed
    :param metric: Metric used for distances of x, must be a metric available for sklearn.metrics.pairwise_distances
    :return: tuple of Pearson and Spearman correlation coefficient
    """

    np.random.seed(seed)
    sample_idx = np.random.randint(len(x), size=sample_size)
    x_sample = x[sample_idx]
    y_sample = y[sample_idx]

    x_dists = pairwise_distances(x_sample, metric=metric).flatten()
    y_dists = pairwise_distances(y_sample, metric="euclidean").flatten()

    pear_r, _ = pearsonr(x_dists, y_dists)
    spear_r, _ = spearmanr(x_dists, y_dists)
    return pear_r, spear_r


def acc_kNN(x, y, k, metric="euclidean"):
    """
    Computes the accuracy of k nearest neighbors between x and y.
    :param x: array of positions for first dataset
    :param y: array of positions for second dataset
    :param k: number of nearest neighbors considered
    :param metric: Metric used for distances of x, must be a metric available for sklearn.metrics.pairwise_distances

    :return: Share of x's k nearest neighbors that are also y's k nearest neighbors
    """

    x_kNN = scipy.sparse.coo_matrix((np.ones(len(x) * k), (
        np.repeat(np.arange(x.shape[0]), k), kNN_graph(x, k, metric=metric).cpu().numpy().flatten())),
                                    shape=(len(x), len(x)))
    y_kNN = scipy.sparse.coo_matrix(
        (np.ones(len(y) * k), (np.repeat(np.arange(y.shape[0]), k), kNN_graph(y, k).cpu().numpy().flatten())),
        shape=(len(y), len(y)))
    overlap = x_kNN.multiply(y_kNN)
    matched_kNNs = overlap.sum()
    return matched_kNNs / (len(x) * k)


def kNN_graph(x, k, metric="euclidean"):
    """
    Pykeops implementation of a k nearest neighbor graph
    :param x: array containing the dataset
    :param k: number of neartest neighbors
    :param metric: Metric used for distances of x, must be "euclidean" or "cosine".
    :return: array of shape (len(x), k) containing the indices of the k nearest neighbors of each datapoint
    """

    dists = keops_dists(x, metric)
    knn_idx = dists.argKmin(K=k + 1, dim=0)[:,
              1:]  # use k+1 neighbours and omit first, which is just the point itself

    return knn_idx


def keops_dists(x, metric):
    """
    Creates a keops lazytensor with the pairwise distances.
    :param x: np.array(n, d) Data points
    :param metric: str The metric used to compute the distance, must be one of "correlation", "euclidean" or "cosine"
    :return: lazytensor (n, n)
    """

    x = torch.tensor(x).to("cuda").contiguous()
    if metric == "correlation":
        # mean center so that we can then do the same thing as for cosine
        x -= x.mean(axis=-1, keepdims=True)
        x_i = LazyTensor(x[:, None])
        x_j = LazyTensor(x[None])
        if metric == "euclidean":
            dists = ((x_i - x_j) ** 2).sum(-1)
        elif metric == "cosine" or metric == "correlation":
            scalar_prod = (x_i * x_j).sum(-1)
            norm_x_i = (x_i ** 2).sum(-1).sqrt()
            norm_x_j = (x_j ** 2).sum(-1).sqrt()
            dists = 1 - scalar_prod / (norm_x_i * norm_x_j)
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented.")
        return dists
