from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'darkblue', 'darkgreen', 'blue', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x1_mesh, x2_mesh = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    Z = Z.reshape(x1_mesh.shape)
    plt.contourf(x1_mesh, x2_mesh, Z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cls in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cls, 0],
                    y=X[y == cls, 1],
                    alpha=0.8,
                    c=cmap(idx),
                    marker=markers[idx],
                    label=cls)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='orange',
                alpha=1.0, linewidth=1, marker='*',
                s=55, label='test set')

def eigenpair(X):
    cov_mat = np.cov(X.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    return eigen_vals, eigen_vecs

def selftest():
    X = (np.random.randn(3, 3) * 100 + 3).round()
    print(X)
    print(eigenpair(X))

if __name__ == '__main__':
    selftest()
