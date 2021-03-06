#!/usr/bin/env python3
# Data helper for NAICS classification
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
import string
import pickle
from ast import literal_eval
import nltk.data;nltk.data.path.append("/media/sf_patsnap/nltk_data")
from nltk import wordpunct_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
import pandas as pd

NAICS_CODES_PATH = "../data/naics/codes_3digits.csv"
MIN_TRAIN_LINE = 128

expected_types = ("NNP", "NN", "NNS", "JJ")

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def tokenize_text(doc):
    poster = PorterStemmer()
    tokens = []
    for w, t in pos_tag(wordpunct_tokenize(clean_str(doc))):
        w = poster.stem(w)
        if t in expected_types and w.isalpha() and w not in stopwords.words('english'):
            tokens.append(w)
    return tokens


def gen_line(fd):
    for line in fd:
        line = literal_eval(line.strip())
        if len(line[0]) < MIN_TRAIN_LINE or line[1] is None:
            continue
        yield line

def reformat(data_path, token_path, ds_path):
    for p in (data_path, token_path, ds_path):
        if not os.path.dirname(p):
            os.makedirs(os.path.dirname(p))

    global_tokens = {}
    with open(data_path) as fd:
        for line in gen_line(fd):
            tokens = tokenize_text(line[0])
#            gen_tokens_from_line(tokens, global_tokens, token_path)
            build_dataset_from_line(line, ds_path)

#    gen_tokens_from_line([], global_tokens, token_path)

def gen_tokens_from_line(tokens, global_tokens, token_path):
    [global_tokens.update({t: len(global_tokens)}) \
            for t in tokens if t not in global_tokens]
    if len(global_tokens) % 100 == 0:
        print("Total tokens: {}".format(len(global_tokens)))
        with open(token_path, 'wb') as fd:
            pickle.dump(global_tokens, fd)

def build_dataset_from_line(line, output):
    clean_text = re.sub(r'(NAICS|#|http.*:/(/\w+)*'+'|'+str(line[1])+')', '', line[0])
    clean_text = ''.join(filter(lambda x: x not in string.punctuation, clean_text))
    clean_text = clean_text[:131000]
    df = pd.DataFrame({"desc":[clean_text],"code":[line[1]]})
    sep = '#'

    if not os.path.isfile(output):
        df.to_csv(output, header=True, index=False, sep=sep)
    else:
        df.to_csv(output, header=False, index=False, sep=sep, mode='a')

def load_d3table(data_path=None):
    if not data_path:
        data_path = NAICS_CODES_PATH
    return pd.read_csv(NAICS_CODES_PATH, engine='python',
            header=0, delimiter="#", dtype={"code":np.int})

def level_decode(index, d3table=None, class_map=None):
    iid = class_map[index]
    return d3table[d3table["code"] == iid].values

def extract_xy(chunk):
    return chunk["desc"], chunk["code"]

def load_class_map():
    chunk = pd.read_csv(NAICS_CODES_PATH, engine='python',
            header=0, delimiter="#", dtype={"code":np.int})
    ret = {}
    chunk.apply(lambda x: ret.update({x["code"]:x["title"]}), axis=1)
    return ret

if __name__ == '__main__':
    data_path = "../data/naics/full.txt"
    token_path = "../data/naics/global_tokens.pickle"
    ds_path = "../data/naics/full.csv"
    reformat(data_path, token_path, ds_path)
#    load_class_map()
