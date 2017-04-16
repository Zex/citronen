from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINE_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

def build_data():
    df_wine = pd.read_csv(WINE_DATA, header=None)
    df_wine.columns = ['Class label', 'Alcohol',
            'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
            'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 'Proanthocyanins',
            'Color intensity', 'Hue',
            'OD280/0D315 of diluted wines', 'Proline']
    print('Class labels:\n', np.unique(df_wine['Class label']))
    print('head:\n', df_wine.head())
    return df_wine

def wine_learn():
    df_wine = build_data()
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.3, random_state=0)
    print(X_train, X_test, y_train, y_test)
    """
    x_norm(i) = (x(i)-x_min)/(x_max-x_min)

    """
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    print(X_train_norm, X_test_norm)
    """
    x_std(i) = (x(i)-x_mu)/x_sigma
    """
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    print('accuracy/train\n:', lr.score(X_train_std, y_train))
    print('accuracy/test\n:', lr.score(X_test_std, y_test))
    print('intercept:\n', lr.intercept_)
    print('coef:\n', lr.coef_)

    colors = ['blue', 'green', 'red', 'cyan',
        'magenta', 'yellow', 'black', 'pink',
        'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

    weights, params = [], []
    for c in np.arange(-4, 6):
        C = 10**c if c > 0 else 1/10**(-c)
        lr = LogisticRegression(penalty='l1', C=C, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(C)
    weights = np.array(weights)
    plotting(df_wine, weights, params, colors)

def plotting(df_wine, weights, params, colors):
    fig = plt.figure()
    ax = plt.subplot(111)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column], label=df_wine.columns[column+1],
                color=color)

    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim(10**(-5), 10**5)
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()

if __name__ == '__main__':
    wine_learn()


