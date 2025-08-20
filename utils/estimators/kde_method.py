import numpy as np
from sklearn.neighbors import KernelDensity
from math import log

def mutual_information_kde(X, Y, bandwidth=1.0, kernel='gaussian'):
    """
    Estimate mutual information I(X;Y) using Kernel Density Estimation (KDE).
    X: array-like, shape (n_samples_x,)
    Y: array-like, shape (n_samples_y,)
    bandwidth: bandwidth parameter for KDE
    kernel: the type of kernel to use ('gaussian' is common)
    """
    X = np.ravel(X)
    Y = np.ravel(Y)

    # Fit KDE for X and Y separately
    kde_X = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde_Y = KernelDensity(kernel=kernel, bandwidth=bandwidth)

    kde_X.fit(X[:, np.newaxis])  # Fit the KDE for X
    kde_Y.fit(Y[:, np.newaxis])  # Fit the KDE for Y

    # Compute log-likelihood for both X and Y
    log_likelihood_X = kde_X.score_samples(X[:, np.newaxis])
    log_likelihood_Y = kde_Y.score_samples(Y[:, np.newaxis])

    # Compute joint log-likelihood
    log_likelihood_joint = kde_X.score_samples(X[:, np.newaxis]) + kde_Y.score_samples(Y[:, np.newaxis])

    # Calculate mutual information from the densities
    MI = np.mean(log_likelihood_joint - (log_likelihood_X + log_likelihood_Y))

    return MI

