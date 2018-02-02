# Data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
import pickle
from enum import Enum, unique
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle


@unique
class Mode(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    EVAL = "EVAL"

class Config(object):
    
    def __init__(self):
        self.batch_size = 32
        self.data_path = 'data/price/train.tsv'
        self.content_path = 'data/price/content.pickle'
        self.mode = Mode.TRAIN
        self.need_shuffle = False

    def is_training(self):
        return self.mode == Mode.TRAIN

    def is_testing(self):
        return self.mode == Mode.TEST

    def is_evaluating(self):
        return self.mode == Mode.EVAL


def preprocess(cfg):
    gen = pd.read_csv(cfg.data_path, delimiter='\t', chunksize=cfg.batch_size)
    yield from [foreach_df(df, cfg) for df in gen]

def foreach_df(df, cfg):
    df = df.drop_duplicates()

    if cfg.need_shuffle:
        df = shuffle(df)

    X = encode_text(df, cfg)
    X = np.concatenate((X, df['shipping'].values.reshape(X.shape[0], 1)), 1).astype(np.float)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    if cfg.is_training():
        target = df['price'].astype(np.float)
        y = target.values.reshape(target.shape[0], 1)
        return X, y
    print(df.keys())
    iid = df['test_id'].values
    return iid, X

def encode_text(df, cfg):
    cate = df['category_name'].fillna('').values
    name = df['name'].fillna('').values
    desc = df['item_description'].fillna('').values
    
    content = list(map(lambda l: ' '.join([l[0], l[1]]), zip(cate, name)))
    content = list(map(lambda l: ' '.join([l[0], l[1]]), zip(content, desc)))
    le, ret = load_or_fit(cfg.content_path, content=content)
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


def to_csv(df, path):
    if not os.path.isfile(path):
        df.to_csv(path, index=None, float_format='%0.6f')
    else:
        df.to_csv(path, index=None, float_format='%0.6f', header=False, mode='a')

def init(cls):
    parser = argparse.ArgumentParser(description="Price estimation")
    parser.add_argument('--train', action='store_true', default=False, help='train a model')
    parser.add_argument('--test', action='store_true', default=False, help='test a model')
    parser.add_argument('--eval', action='store_true', default=False, help='eval a model')
    parser.add_argument('--anal', action='store_true', default=False, help='analyse data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000000, help='total epochs')
    parser.add_argument('--load_model', action='store_true', default=False, help='load exist model')
    parser.add_argument('--model_dir', type=str, default='models/price', help='model directory')
    parser.add_argument('--lr', type=float, default=1e-10, help='initial learning rate')
    parser.add_argument('--summ_intv', type=int, default=1000, help='summary interval')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
    return parser.parse_args()
