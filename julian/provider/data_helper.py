#!/usr/bin/env python3
# Data helper 
import os
import re
import pickle
import pandas as pd
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
# Use the correct path if not the default ones
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

expected_types = ("NNP", "NN", "NNS", "JJ")


def persist(obj, path):
    with open(path, 'w+b') as fd:
        pickle.dump(obj, fd)


def from_persist(path):
    with open(path, 'r+b') as fd:
        ret = pickle.load(fd)
    return ret


def clean_lang(data_path, output_dir):
    reader = pd.read_csv(data_path, engine='python', header=0, 
        delimiter="#", chunksize=1)

    for chunk in reader:
        text, l1, l2 = extract_xy(chunk)
        text, l1, l2 = text.values[0], l1.values[0], l2.values[0]
        ratio = guess_lang(text)
        lang = max(ratio, key=ratio.get)
        output = "{}/{}.csv".format(output_dir, lang)
        df = pd.DataFrame({"desc":[text],"cate":[l1],"subcate":[l2]})

        if not os.path.isfile(output):
            df.to_csv(output, header=True, index=False, sep="#")
        else:
            df.to_csv(output, header=False, index=False, sep="#", mode='a')


def guess_lang(text):
    ratio = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    for lang in stopwords.fileids():
        stopwords_set = set(stopwords.words(lang))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        ratio[lang] = len(common_elements)
   
    return ratio


def tokenize_text(text):
    poster = PorterStemmer()
    tokens = []
    for doc in text:
        for w, t in pos_tag(wordpunct_tokenize(clean_str(doc))):
            w = poster.stem(w)
            if t in expected_types and w.isalpha() and w not in stopwords.words('english'):
                tokens.append(w)
    return tokens


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


def get_hashing_vec(max_doc=5000, stop_words="english"):
    return HashingVectorizer(
                ngram_range=(1,5),
                stop_words=stop_words,
                n_features=self.max_doc,
                tokenizer=nltk.word_tokenize,
                dtype=np.int32,
                analyzer='word')
