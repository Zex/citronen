# Data provider 
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
from tensorflow.contrib import learn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class Provider(object):
    
    def __init__(self, args, need_shuffle=True):
        self.max_doc = args.max_doc
        self.need_shuffle = need_shuffle
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.vocab_path = args.vocab_path if args.vocab_path \
                else os.path.join(args.model_dir,
                "vocab_{}th".format(datetime.today().timetuple().tm_yday))

    def load_city_data(self):
        chunk = pd.read_csv(self.data_path, header=0, delimiter=",")
        if self.need_shuffle:
            chunk = shuffle(chunk)
        # country_iso_code,country_name, city_name
        one = chunk['country_iso_code'].to_frame().rename(\
                columns={'country_iso_code':'vocab'}).append([\
                chunk['country_name'].to_frame().rename(columns={'country_name':'vocab'}),
                chunk['city_name'].to_frame().rename(columns={'city_name':'vocab'})])
        one = one.dropna()
        return one

    def train_vocab_from_data(self, chunk):
        vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_doc)
        x = list(vocab_processor.fit_transform(chunk))
        print("vocab size", len(vocab_processor.vocabulary_))
    
        if self.vocab_path:
            dirpath = os.path.dirname(self.vocab_path)
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            vocab_processor.save(self.vocab_path)
        return vocab_processor

    def train_vocab(self):
        data = self.load_city_data()
        self.train_vocab_from_data(data['vocab'])

    def load_vocab(self):
        if not os.path.isfile(self.vocab_path):
            self.train_vocab()
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)

    def load_data(self):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        chunk = shuffle(chunk)
        return np.array(list(
            self.vocab_processor.transform(
                chunk["full_name"])), dtype=np.float32), None

    def batch_data(self):
        x, _ = self.load_data()
        total_batch = int((len(x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield x[current:current+self.batch_size+1]
