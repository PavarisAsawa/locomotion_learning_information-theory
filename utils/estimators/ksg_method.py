
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import log

def _entropy_knn(X, k=3):
    """
    Helper function: Estimate differential entropy using KNN.
    """
    n, d = X.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # kth nearest neighbor distance (exclude the point itself)
    eps = distances[:, k] + 1e-15
    volume_unit_ball = (np.pi ** (d / 2)) / np.math.gamma(d / 2 + 1)
    h = (d * np.mean(np.log(eps)) 
         + log(volume_unit_ball) 
         + log(n - 1)
         - log(k))
    return h

def mutual_information_ksg(X, Y, k=3):
    """
    Estimate mutual information I(X;Y) using KSG estimator (KNN-based).
    X: array-like, shape (n_samples, n_features_X)
    Y: array-like, shape (n_samples, n_features_Y)
    k: number of neighbors
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    XY = np.hstack([X, Y])
    H_X = _entropy_knn(X, k)
    H_Y = _entropy_knn(Y, k)
    H_XY = _entropy_knn(XY, k)
    MI = H_X + H_Y - H_XY
    return MI

def mutual_information_ksg_(X, Y, k=3):
    """
    Estimate mutual information between two datasets using the Kraskov, Stögbauer, and Grassberger (KSG) estimator.

    Parameters:
        X (ndarray): 2D array of shape (N, dX), where N is the number of samples and dX is the number of features for X.
        Y (ndarray): 2D array of shape (N, dY), where N is the number of samples and dY is the number of features for Y.
        k (int): The number of nearest neighbors to use.

    Returns:
        float: The estimated mutual information I(X, Y).
    """
    
    # Combined dataset for distance computation
    Z = np.hstack((X, Y))  # Join X and Y into a single array

    # Use NearestNeighbors to compute distances
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(Z)
    distances, indices = nbrs.kneighbors(Z)

    # Compute the distances for X and Y subspaces
    ex = np.linalg.norm(X[indices[:, 1:]] - X[indices[:, 0]].reshape(-1, 1), axis=2)
    ey = np.linalg.norm(Y[indices[:, 1:]] - Y[indices[:, 0]].reshape(-1, 1), axis=2)
    ez = np.linalg.norm(Z[indices[:, 1:]] - Z[indices[:, 0]].reshape(-1, 1), axis=2)
    
    # Compute the quantities for the KSG estimator
    term1 = np.mean(np.log(ez[:, k-1])) - np.mean(np.log(ex[:, k-1])) - np.mean(np.log(ey[:, k-1]))
    term2 = digamma(k) - digamma(len(X))
    
    mutual_info = term1 + term2
    return mutual_info

