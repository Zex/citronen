# Data provider
#
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import pickle


BATCH_SIZE = 512
path = 'data/price/train.tsv'

def preprocess(need_shuffle=True, mode='TRAIN'):
    global path
    batch_size = BATCH_SIZE
    gen = pd.read_csv(path, delimiter='\t', chunksize=batch_size)
    yield from [foreach_df(df, need_shuffle, mode) for df in gen]


def foreach_df(df, need_shuffle=True, mode="TRAIN"):
    df = df.drop_duplicates()
    if need_shuffle:
        df = shuffle(df)

    X = encode_text(df)
    X = np.concatenate((X, df['shipping'].values.reshape(X.shape[0], 1)), 1).astype(np.float)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    if mode == 'TRAIN':
        target = df['price'].astype(np.float)
        y = target.values.reshape(target.shape[0], 1)
        return X, y

    iid = df['test_id'].values
    return iid, X

def encode_text(df):
    path = 'data/price/content.pickle'

    cate = df['category_name'].fillna('').values
    name = df['name'].fillna('').values
    desc = df['item_description'].fillna('').values
    
    content = list(map(lambda l: ' '.join([l[0], l[1]]), zip(cate, name)))
    content = list(map(lambda l: ' '.join([l[0], l[1]]), zip(content, desc)))
    le, ret = load_or_fit(path, content=content)
    return ret


def load_or_fit(path, df=None, field=None, content=None):
    if field and df:
        input_x = df[field].fillna('')

    if content:
        input_x = content

    if not os.path.isfile(path):
        le = TfidfVectorizer()
        le = le.fit(input_x)

        with open(path, 'wb') as fd:
            pickle.dump(le, fd)
    else:
        with open(path, 'rb') as fd:
            le = pickle.load(fd)
    
    ret = le.transform(input_x).toarray()
    return le, ret
