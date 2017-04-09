import numpy as np
import pandas
import matplotlib.pyplot as plt

class AdalineGD(object):
    """
    Sum Squared Error(SSE)
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            print("w: {}, cost: {}".format(self.w_, cost))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.netinput(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

def selftest():
    X, y = np.random.randn(1000, 1000), np.arange(1000)

    epoch1, lr1 = 100, 0.01
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ada1 = AdalineGD(n_iter=epoch1, eta=lr1).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(SSE)')
    ax[0].set_title('Adaline - Learning rate {}'.format(lr1))

    epoch2, lr2 = 100, 0.0001
    ada2 = AdalineGD(n_iter=epoch2, eta=lr2).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(SSE)')
    ax[1].set_title('Adaline - Learning rate {}'.format(lr2))

    plt.show()
    
if __name__ == "__main__":
    selftest()
