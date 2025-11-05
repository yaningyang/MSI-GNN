import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def estimate_gamma(X):
    # compute pairwise squared distances upper triangle median
    from scipy.spatial.distance import pdist
    dists = pdist(X, metric='sqeuclidean')
    med = np.median(dists[dists > 0]) if dists.size>0 else 1.0
    sigma = np.sqrt(med) if med > 0 else 1.0
    gamma = 1.0 / (2 * (sigma ** 2) + 1e-12)
    return gamma

def build_rbf_knn_graph(X, k=8, gamma=None, feature_weights=None, keep_self=False):
    """
    Build sparse kNN graph using RBF kernel similarities.
    Returns edge_index (2, E) torch.long and edge_weight (E,) torch.float.
    """
    N, D = X.shape
    Xw = X.copy()
    if feature_weights is not None:
        feature_weights = np.asarray(feature_weights).reshape(1, -1)
        Xw = Xw * feature_weights

    if gamma is None:
        gamma = estimate_gamma(Xw)

    # compute k nearest neighbors by Euclidean in weighted space (efficient)
    nbrs = NearestNeighbors(n_neighbors=min(k+1, N), algorithm='auto', metric='euclidean').fit(Xw)
    distances, indices = nbrs.kneighbors(Xw)  # includes self as first neighbor
    rows = []
    cols = []
    weights = []
    for i in range(N):
        for jj, j in enumerate(indices[i]):
            if (not keep_self) and (j == i):
                continue
            # compute RBF similarity from squared euclidean dist
            sqdist = np.sum((Xw[i] - Xw[j])**2)
            sim = np.exp(-gamma * sqdist)
            rows.append(i)
            cols.append(j)
            weights.append(sim)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight, gamma
