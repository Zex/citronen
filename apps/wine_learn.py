from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from apps.utils import eigenpair, plot_decision_regions
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
    with_pca(X_train_std, y_train, X_test_std)
    plotting(df_wine, weights, params, colors)
    with_randomforest(X_train, y_train, df_wine.columns[1:])
    with_sbs(X_train_std, y_train)

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
    plt.close()

def with_sbs(X_train_std, y_train):
    from apps.sbs import SBS
    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    k_feat = [len(k) for k in sbs.subsets_]
    
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()
    plt.close()

def with_randomforest(X_train, y_train, feat_labels):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10000,
            random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print(f+1, 30, feat_labels[f], importances[indices[f]])
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices],
            color='blue', align='center')
    plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()
    plt.close()

def with_pca(X_train_std, y_train, X_test_std):
    eigen_vals, eigen_vecs = eigenpair(X_train_std)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w = np.hstack((
        eigen_pairs[0][1][:, np.newaxis],
        eigen_pairs[1][1][:, np.newaxis]))
    X_train_pca = X_train_std[0].dot(w)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='best')
    plt.show()
    plt.close()

if __name__ == '__main__':
    wine_learn()
    


