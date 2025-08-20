
import numpy as np
from math import log
from itertools import product

def mutual_information_binning(X, Y, bins=10):
    """
    Estimate mutual information I(X;Y) using histogram/binning method.
    X: array-like, shape (n_samples,)
    Y: array-like, shape (n_samples,)
    bins: number of bins
    """
    X = np.ravel(X)
    Y = np.ravel(Y)
    
    # Joint histogram
    joint_hist, x_edges, y_edges = np.histogram2d(X, Y, bins=bins)
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Marginals
    px = np.sum(joint_prob, axis=1)
    py = np.sum(joint_prob, axis=0)
    
    # Compute mutual information
    MI = 0.0
    for i, j in product(range(len(px)), range(len(py))):
        pxy = joint_prob[i, j]
        if pxy > 0:
            MI += pxy * log(pxy / (px[i] * py[j]))
    return MI
