import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from apps.utils import plot_decision_regions

def logistic_regression(X_train_std, y_train, X_test_std, y_test):
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    X_comb_std = np.vstack((X_train_std, X_test_std))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(X_comb_std, y_comb, classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def decision_regions(X_train_std, y_train, X_test_std, y_test, classifier):
    X_comb_std = np.vstack((X_train_std, X_test_std))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(X_comb_std, y_comb, classifier=classifier, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

def selftest():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test!=y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    decision_regions(X_train_std, y_train, X_test_std, y_test, ppn)
    logistic_regression(X_train_std, y_train, X_test_std, y_test)

if __name__ == '__main__':
    selftest()

