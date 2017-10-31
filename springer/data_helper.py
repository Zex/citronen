#!/usr/bin/env python3
# Data helper 
import pandas as pd
import os
import re
import pickle
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk.data;nltk.data.path.append("/media/sf_patsnap/nltk_data")
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from tensorflow.contrib import learn
from nltk.stem.porter import PorterStemmer

GLOBAL_TOKENS_PATH = "../data/springer/lang/token_english.pickle"
L1_TABLE_PATH = "../data/springer/l1_table.pickle"
L2_TABLE_PATH = "../data/springer/l2_table.pickle"

l1_table = {}
l2_table = {}

expected_types = ("NNP", "NN", "NNS", "JJ")

def persist(obj, path):
    with open(path, 'w+b') as fd:
        pickle.dump(obj, fd)

def from_persist(path):
    with open(path, 'r+b') as fd:
        ret = pickle.load(fd)
    return ret

def encode(data_path):

    def assign_l1(cate):
        #cate = cate.replace("\\/", "/").replace("\/", "/")
        if cate not in l1_table:
            l1_table.update({cate:0x1000 if not l1_table else max(l1_table.values()) + 0x1000})

    reader = pd.read_csv(data_path, engine='python', header=0, chunksize=100, delimiter="###")
    for chunk in reader:
        chunk["cate"].apply(assign_l1)
    
    persist(l1_table, L1_TABLE_PATH)

    mem = {}
    def assign_l2(cate, subcate):
        #cate = cate.replace("\\/", "/").replace("\/", "/")
        #subcate = subcate.replace("\\/", "/").replace("\/", "/")
        if subcate not in l2_table:
            l2_table.update({subcate:l1_table[cate]+1 if cate not in mem else mem[cate]+1})
            mem.update({cate:l2_table[subcate]})

    reader = pd.read_csv(data_path, engine='python', header=0, chunksize=100, delimiter="###")
    for chunk in reader:
        chunk.apply(lambda x: assign_l2(x["cate"], x["subcate"]), axis=1)
    persist(l2_table, L2_TABLE_PATH)

def load_l2table():
    return from_persist(L2_TABLE_PATH)

def load_l1table():
    return from_persist(L1_TABLE_PATH)
    
def extract_xy(chunk, l1table=None, l2table=None):
    chunk = chunk.dropna()

    if l1table:
        chunk = chunk.replace({"cate":l1table})
    if l2table:
        chunk = chunk.replace({"subcate":l2table})

    text = chunk["desc"]
    label1 = chunk["cate"]
    label2 = chunk["subcate"]
    return text, label1, label2

def clean_lang(data_path):
    reader = pd.read_csv(data_path, engine='python', header=0, 
        delimiter="###", chunksize=1)

    for chunk in reader:
        text, l1, l2 = extract_xy(chunk)
        text, l1, l2 = text.values[0], l1.values[0], l2.values[0]
        ratio = guess_lang(text)
        lang = max(ratio, key=ratio.get)
        output = "../data/springer/lang/{}.csv".format(lang)
        df = pd.DataFrame({"desc":[text],"cate":[l1],"subcate":[l2]})

        if not os.path.isfile(output):
            df.to_csv(output, header=True, index=False, sep="###")
        else:
            df.to_csv(output, header=False, index=False, sep="###", mode='a')

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

def train_vocab(data_path, vocab_path=None, max_doc_len=50000):
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_len)
    reader = pd.read_csv(data_path, engine="python", 
            header=0, delimiter="###", chunksize=512)
    for chunk in reader:
        text, _, l2 = extract_xy(chunk)
        tokens = tokenize_text(text)
        vocab_processor.fit([' '.join(t) for t in tokens])
        print("vocab size", len(vocab_processor.vocabulary_))
        break

    if vocab_path:
        dirpath = os.path.dirname(vocab_path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        vocab_processor.save(vocab_path)
    return vocab_processor

class VocabProcessor(object):

    def __init__(self, max_document_length, min_frequency=0):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequence

    def fit(self, raw_documents, unused_y=None):
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        #self.vocabulary_.freeze()
        return self
    
    def transform(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

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

def gen_token(data_path, output):
    chunksize = 512
    reader = pd.read_csv(data_path, engine="python", 
            header=0, delimiter="###", chunksize=chunksize)

    l2_table = load_l2table()
    global_tokens = {}

    if not os.path.dirname(output):
        os.makedirs(os.path.dirname(output))

    for chunk in reader:
        text, _, l2 = extract_xy(chunk, l2table=l2_table)
        tokens = tokenize_text(text)
        [global_tokens.update({t: len(global_tokens)}) for t in tokens if t not in global_tokens]
        
        if len(global_tokens) % 100 == 0:
            with open(output, 'w+b') as fd:
                pickle.dump(global_tokens, fd)

    print("global tokens", len(global_tokens))

def load_global_tokens():
    with open(GLOBAL_TOKENS_PATH, 'rb') as fd:
        global_tokens = pickle.load(fd)
    return global_tokens

def text2vec(chunk, l2_table, global_tokens):
    x = []
    text, _, l2 = extract_xy(chunk, l2table=l2_table)
    tokens = tokenize_text(text)
    [x.append(global_tokens[t]) for t in text if t in global_tokens]
    return x


if __name__ == "__main__":
    """
    print(len(l1_table), len(l2_table))
    l2_table = load_l2table()
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    """
    data_path = "../data/springer/full.csv"
    data_path = "../data/springer/lang/english.csv"
    gen_token(data_path, GLOBAL_TOKENS_PATH)
#    simple_encode(data_path)
#    clean_lang(data_path)

