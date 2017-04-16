import numpy as np
import matplotlib.pyplot as plt
from apps.utils import eigenpair

def plot_eigen():
    X = (np.random.randn(10, 10) * 100 + 17).round()
    eig_vals, eig_vecs = eigenpair(X)
    tot = sum(eig_vals)
    var_exp = [(i/tot) for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plt.bar(range(10), var_exp, alpha=0.5, align='center',
            label='var variance')
    plt.step(range(10), cum_var_exp, where='mid',
            label='cum variant')
    plt.ylabel('variance ratio')
    plt.xlabel('component')
    plt.legend(loc='best')
    plt.show()

plot_eigen()

