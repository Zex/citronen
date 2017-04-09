import numpy as np
import pandas
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions
from apps.utils import plot_decision_regions

class AdalineGD(object):
    """
    Adaptive Linear neuron classifier
    """
    def __init__(self, lr=0.01, epoch=50):
        """
        @lr Learning rate
        @epoch Number of iteration
        """
        self.lr = lr
        self.epoch = epoch

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epoch):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.lr * X.T.dot(errors)
            self.w_[0] += self.lr * errors.sum()
            cost = (errors**2).sum() / 2.0
            print("w: {}, cost: {}".format(self.w_, cost))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

def selftest():
    """
    SSE - Sum Squared Error
    """
    X, y = np.random.randn(1000, 1000), np.arange(1000)
    # ada1
    epoch1, lr1 = 100, 0.01
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdalineGD(epoch=epoch1, lr=lr1).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SSE)')
    ax[0].set_title('Adaline - Learning rate {}'.format(lr1))
    # ada2
    epoch2, lr2 = 100, 0.0001
    ada2 = AdalineGD(epoch=epoch2, lr=lr2).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(SSE)')
    ax[1].set_title('Adaline - Learning rate {}'.format(lr2))
    plt.show()
    plt.close()
    # ada3
    epoch3, lr3 = 15, 0.01
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
    ada3 = AdalineGD(epoch=epoch3, lr=lr3)
    ada3.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada3)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()
    plt.plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('SSE')

    plt.show()
    plt.close()
    
if __name__ == "__main__":
    selftest()
