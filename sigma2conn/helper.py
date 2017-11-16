# Gaussian plot
# Author: Zex Li <top_zlynch@yahoo.com>
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def dim1_gaussian(sigma, x):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x,2)/(2*np.power(sigma,2)))

def dim2_gaussian(sigma, x, y):
    return 1/(2*np.pi*np.power(sigma,2))*np.exp(-(np.power(x,2)+np.power(y,2))/(2*np.power(sigma,2)))

def l1_norm(y1, y2):
    return np.sum(np.abs(y2-y1))

def l2_norm(y1, y2):
    return np.sum(np.power(y1-y2, 2))

if __name__ == '__main__':
    sigma = 1e-15
    sigma = 1.0
    sigma = 0.82

    n_sample = 1000
    x = np.random.randn(n_sample)
    y = np.vectorize(partial(dim1_gaussian, sigma))(x)
    sns.barplot(x, y)
    plt.show()

    n_sample = 1000
    x = np.random.randn(n_sample)
    y = np.random.randn(n_sample)
    z = np.vectorize(lambda xx,yy: dim2_gaussian(sigma, xx, yy))(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z)
    plt.show()
