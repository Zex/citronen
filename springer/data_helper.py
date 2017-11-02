#!/usr/bin/env python3
# Data helper 
import os
import re
import pickle
import pandas as pd
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
# Use the correct path if not the default ones
import nltk.data;nltk.data.path.append("/media/sf_patsnap/nltk_data")
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from tensorflow.contrib import learn

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

def level_encode(data_path):
    """Encode L1/L2
    """
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

def level_decode(index, l1table=None, l2table=None, class_map=None):
    """Reversed index L1/L2 from class map
    """
    iid, l1name, l2name = None, None, None
    if l2table:
        if not class_map:
            class_map = list(set(l2table.values()))
        iid = class_map[index]
        l2name = dict(map(reversed, l2table.items())).get(iid)
        if l1table:
            l1name = dict(map(reversed, l1table.items())).get(iid//0x1000*0x1000)
    return iid, l1name, l2name

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

def clean_lang(data_path, output_dir):
    reader = pd.read_csv(data_path, engine='python', header=0, 
        delimiter="###", chunksize=1)

    for chunk in reader:
        text, l1, l2 = extract_xy(chunk)
        text, l1, l2 = text.values[0], l1.values[0], l2.values[0]
        ratio = guess_lang(text)
        lang = max(ratio, key=ratio.get)
        output = "{}/{}.csv".format(output_dir, lang)
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

    if vocab_path:
        dirpath = os.path.dirname(vocab_path)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        vocab_processor.save(vocab_path)
    return vocab_processor

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

    l2_table = from_persist(L2_TABLE_PATH)
    global_tokens = {}

    if not os.path.dirname(output):
        os.makedirs(os.path.dirname(output))

    for chunk in reader:
        text, _, l2 = extract_xy(chunk, l2table=l2_table)
        tokens = tokenize_text(text)
        [global_tokens.update({t: len(global_tokens)}) for t in tokens if t not in global_tokens]
        
        if len(global_tokens) % 100 == 0:
            print("Total tokens: {}".format(len(global_tokens)))
            with open(output, 'wb') as fd:
                pickle.dump(global_tokens, fd)

    print("Total tokens: {}".format(len(global_tokens)))
    with open(output, 'wb') as fd:
        pickle.dump(global_tokens, fd)

def load_global_tokens():
    with open(GLOBAL_TOKENS_PATH, 'rb') as fd:
        global_tokens = pickle.load(fd)
    return global_tokens

def get_hashing_vec(max_doc=5000, stop_words="english"):
    return HashingVectorizer(
                ngram_range=(1,5),
                stop_words=stop_words,
                n_features=self.max_doc,
                tokenizer=nltk.word_tokenize,
                dtype=np.int32,
                analyzer='word')

def text2vec(docs, global_tokens, max_doc_len=None):
    ret = []
    for text in docs:
        x = []
        tokens = tokenize_text(text)
        if max_doc_len:
            [x.append(global_tokens[t]) for t in tokens if t in global_tokens and len(x) < max_doc_len]
            x.extend([0]*(max_doc_len-len(x)))
        else:
            [x.append(global_tokens[t]) for t in tokens if t in global_tokens]
        ret.append(x)
    return ret

"""
class TinyVocab(object):

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
"""

def init():
    parser = ArgumentParser()
    parser.add_argument('--data_path', default="../data/springer/full.csv",
            type=str, help='Path to input data')
    parser.add_argument('--output_dir', default="../data/springer/lang",
            type=str, help='Path to output directory')
    parser.add_argument('--table', default=False,
            action="store_true", help='Generate L1/L2 table from data')
    parser.add_argument('--clean_lang', default=False,
            action="store_true", help="Filter data by language")
    parser.add_argument('--gen_token', default=False,
            action="store_true", help="Filter data by language")
    parser.add_argument('--load_table', default=False,
            action="store_true", help='Load L1/L2 table from path')

    args = parser.parse_args()
    return args

def start():
    args = init()
    if args.table:
        level_encode()
    if args.clean_lang:
        clean_lang(args.data_path, args.output_dir)
    if args.gen_token:
        gen_token(args.data_path, GLOBAL_TOKENS_PATH)
    if args.load_table:
        table = from_persist(args.data_path)
        print(len(table))

if __name__ == "__main__":
    from argparse import ArgumentParser
    start()

