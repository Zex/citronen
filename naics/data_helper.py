#!/usr/bin/env python3
# Data helper for NAICS classification
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
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

def reformat(data_path, token_path, ds_path):
    for p in (data_path, token_path, ds_path):
        if not os.path.dirname(p):
            os.makedirs(os.path.dirname(p))

    global_tokens = {}
    with open(data_path) as fd:
        #open(ds_path, 'w+') as ds:
        #ds.write("desc,target\n")
        for line in fd:
            line = literal_eval(line.strip())
            if len(line[0]) < MIN_TRAIN_LINE or line[1] is None:
                continue
            tokens = tokenize_text(line[0])
            gen_tokens_from_line(tokens, global_tokens, token_path)

            #[global_tokens.update({t: len(global_tokens)}) \
            #    for t in tokens if t not in global_tokens]
            #build_dataset_from_line(tokens, int(re.search('\d+', line[1].strip()).group()), global_tokens, ds)
    gen_tokens_from_line([], global_tokens, token_path)

def gen_tokens_from_line(tokens, global_tokens, token_path):
    [global_tokens.update({t: len(global_tokens)}) \
            for t in tokens if t not in global_tokens]
    if len(global_tokens) % 100 == 0:
        print("Total tokens: {}".format(len(global_tokens)))
        with open(token_path, 'wb') as fd:
            pickle.dump(global_tokens, fd)

def build_dataset_from_line(tokens, target, global_tokens, fd):
    for i, t in enumerate(tokens):
        fd.write(str(global_tokens[t]))
        if i < len(tokens)-1:
            fd.write(' ')
    fd.write(",{}\n".format(target))

def load_class_map():
    chunk = pd.read_csv(NAICS_CODES_PATH, engine='python',
            header=0, delimiter="#", dtype={"code":np.int})
    ret = {}
    chunk.apply(lambda x: ret.update({x["code"]:x["title"]}), axis=1)
    return ret

if __name__ == '__main__':
    data_path = "../data/naics/full.txt"
    token_path = "../data/naics/global_tokens.pickle"
    ds_path = "../data/naics/dataset.csv"
    reformat(data_path, token_path, token_path)
#    load_class_map()
