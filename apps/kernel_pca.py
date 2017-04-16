import numpy as np
import matplotlib.pyplot as plt
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA
    """
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma*mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    X_pc = np.column_stack((eigvecs[:, -i] for i in range(1, n_components+1)))
    return X_pc

def half_moon():
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.show()


half_moon()

