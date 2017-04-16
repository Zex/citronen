import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def map_label(df):
    # string => int
    class_mapping = {
        label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
    df['classlabel'] = df['classlabel'].map(class_mapping)
    # int => string
    inv_class_mapping = {
            v: k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print('='*80)
    print(df)
    
def create_label():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1'],
        ])
    df.columns = ['color', 'size', 'price', 'classlabel']
    size_mapping = {
            'XL': 3,
            'L': 2,
            'M': 1
            }
    df['size'] = df['size'].map(size_mapping)
    return df

def one_hot_encode(X):
    ohe = OneHotEncoder(categorical_features=[0])
    return ohe.fit_transform(X).toarray()

def encode_label(df):
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    class_le.inverse_transform(y)
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    return X, y

def selftest():
    df = create_label()
    print('df:', df)
    map_label(df)
    print('='*80)
    print('df:', df)
    X, y = encode_label(df)
    print('='*80)
    print('X:', X)
    print('y:', y)
    X = one_hot_encode(X)
    print('='*80)
    print('X:', X)
    print('dummy features:\n', pd.get_dummies(df[['price', 'color', 'size']]))
    
if __name__ == '__main__':
    selftest()

