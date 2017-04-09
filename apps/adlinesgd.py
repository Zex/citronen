import numpy as np
#from mlxtend.plotting import plot_decision_regions
from apps.utils import plot_decision_regions

class AdalineSGD(object):
    """
    Adaptive Linear neuron classifier with Stochastic Gradient Descent
    """
    def __init__(self, lr=0.01, epoch=10,
            shuffle=True, random_state=None):
        self.lr = lr
        self.epoch = epoch
        self.w_initialized = False
        self.shuffle = shuffle
        np.random.seed(random_state) if random_state else None

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialized_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.epoch):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1+m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.lr * xi.dot(error)
        self.w_[0] += self.lr * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

def selftest():

    X, y = np.random.randn(1000, 1000), np.arange(1000)

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()
    ada = AdalineSGD(epoch=15, lr=0.01, random_state=1)
    ada.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Stochatic Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.plot(range(1, len(ada3.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Cost')
    plt.show()

if __name__ == '__main__':
    selftest()
