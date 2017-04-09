##
# PYTHONPATH=. python3 apps/train_test.py
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    plt.close()

def decision_regions(X_train_std, y_train, X_test_std, y_test, classifier):
    X_comb_std = np.vstack((X_train_std, X_test_std))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(X_comb_std, y_comb, classifier=classifier, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

def logistic_regression_ovr(X_train_std, y_train, X_test_std, y_test):
    weights, params = [], []
    for c in np.arange(-5, 5): # !! pydata/numexpr/issues/272
        C = 10**c
        lr = LogisticRegression(C=C, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(C)
    weights = np.array(weights)
    plt.plot(params, weights[:, 0], label='petal length')
    plt.plot(params, weights[:, 1], linestyle='-', label='petal width')
    plt.xlabel('C')
    plt.ylabel('weight coefficient')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.show()
    plt.close()

def linear_kernel(X_train_std, y_train, X_test_std, y_test):
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    X_comb_std = np.vstack((X_train_std, X_test_std))
    y_comb = np.hstack((y_train, y_test))
    plot_decision_regions(X_comb_std, y_comb,
            classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()
    
def selftest():
    """
Stack
------
>>> x
array([[ 0.17842811, -0.86856259, -1.02279844,  0.63604641],
       [-0.09077123,  0.92990947,  0.01011034, -1.48000142],
       [-1.67795357,  1.74202317,  0.35183575,  0.89971502]])
>>> v
array([[-0.23556223,  1.56660423,  0.25119982, -1.70008857],
       [-0.90285085, -1.68173916,  0.42380589, -0.81406412],
       [ 0.68886529, -0.38166227,  0.28694076, -1.69478095]])
>>> np.hstack((x,v))
array([[ 0.17842811, -0.86856259, -1.02279844,  0.63604641, -0.23556223,
         1.56660423,  0.25119982, -1.70008857],
       [-0.09077123,  0.92990947,  0.01011034, -1.48000142, -0.90285085,
        -1.68173916,  0.42380589, -0.81406412],
       [-1.67795357,  1.74202317,  0.35183575,  0.89971502,  0.68886529,
        -0.38166227,  0.28694076, -1.69478095]])
>>> np.vstack((x,v))
array([[ 0.17842811, -0.86856259, -1.02279844,  0.63604641],
       [-0.09077123,  0.92990947,  0.01011034, -1.48000142],
       [-1.67795357,  1.74202317,  0.35183575,  0.89971502],
       [-0.23556223,  1.56660423,  0.25119982, -1.70008857],
       [-0.90285085, -1.68173916,  0.42380589, -0.81406412],
       [ 0.68886529, -0.38166227,  0.28694076, -1.69478095]])

Using sklearn classifier
------------------------
from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')
    """
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
#    logistic_regression_ovr(X_train_std, y_train, X_test_std, y_test)
    linear_kernel(X_train_std, y_train, X_test_std, y_test)

if __name__ == '__main__':
    selftest()
