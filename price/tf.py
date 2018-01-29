#
#
import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def preprocess():
    path = 'data/price/train.tsv'
    df = pd.read_csv(path, delimiter='\t')
    #df = df.dropna()
    df = df.drop_duplicates()
    
    encode_cate(df)
    encode_name(df)
    encode_desc(df)

def encode_cate(df):
    path = 'data/price/cate.pickle'
    le, grp, cate = load_or_fit(df, path, 'category_name')

    if not grp:
        return

    print('='*30)
    for n, c in zip(grp.groups.keys(), cate):
        print(n, c, le.inverse_transform(c))
        
def encode_name(df):
    path = 'data/price/name.pickle'
    le, grp, name = load_or_fit(df, path, 'name')
   
    if not grp:
        return

    print('='*30)
    for n, c in zip(grp.groups.keys(), name):
        print(n, c, le.inverse_transform(c))

def encode_desc(df):
    path = 'data/price/desc.pickle'
    le, grp, desc = load_or_fit(df, path, 'desc')

    if not grp:
        return
    print('='*30)
    for n, c in zip(grp.groups.keys(), desc):
        print(n, c, le.inverse_transform(c))

def load_or_fit(df, path, field):
    grp = None

    if not os.path.isfile(path):
        grp = df.groupby(field) 
        le = TfidfVectorizer()
        le = le.fit(grp.groups.keys())

        with open(path, 'wb') as fd:
            pickle.dump(le, fd)
    else:
        with open(path, 'rb') as fd:
            le = pickle.load(fd)
    
    return le, grp, le.transform(grp.groups.keys()).toarray()


if __name__ == '__main__':
    preprocess()
