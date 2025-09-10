import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import psi

def ksg_estimator(X, Y, k):
    N = len(X)
    
    # Combine X and Y into a joint dataset Z
    Z = np.column_stack((X, Y))
    
    # Calculate pairwise distances in the joint space and in the marginal spaces
    dist_X = cdist(X, X, metric='euclidean')
    dist_Y = cdist(Y, Y, metric='euclidean')

    # Sort the distances and find the k-th nearest neighbor distances
    dist_X_sorted = np.sort(dist_X, axis=1)
    dist_Y_sorted = np.sort(dist_Y, axis=1)
    # Get the k-th neighbor distances for joint and marginal spaces
    eps_X = dist_X_sorted[:, k]
    eps_Y = dist_Y_sorted[:, k]
    eps_max = np.maximum(eps_X, eps_Y)
    
    # Count the number of neighbors within the k-th neighbor distance for X and Y
    nx = np.array([np.sum(dist_X[i] <= eps_max[i]) for i in range(N)])
    ny = np.array([np.sum(dist_Y[i] <= eps_max[i]) for i in range(N)])
    
    # Compute the digamma functions and the final MI estimate
    digamma_k = psi(k)
    digamma_N = psi(N)
    
    # Average over all data points
    avg_digamma_nx_ny = np.mean(psi(nx + 1) + psi(ny + 1))
    
    # MI estimate
    MI = digamma_k - avg_digamma_nx_ny + digamma_N
    return MI