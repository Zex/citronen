# Data provider for model
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import tensorflow as tf
from julian.provider.data_helper import persist, from_persist


class DataProvider(object):
    """Help prepare data for training, validation and prediction"""
    def __init__(self, args, need_shuffle=True):
        super(DataProvider, self).__init__()
        self.data_path = args.data_path
        self.pred_output = args.pred_output_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.max_doc = args.max_doc
        self.need_shuffle = need_shuffle
        self.vocab_path = args.vocab_path if args.vocab_path \
                else os.path.join(args.model_dir,
                "vocab_{}th".format(datetime.today().timetuple().tm_yday))

    def init_label(self, lbl):
        one = np.zeros(len(self.class_map))
        one[self.class_map.index(lbl)] = 1.
        return one 

    def _process_chunk(self, text, label):
        x = list(self.vocab_processor.transform(text))
        y = list(map(self.init_label, label))
        return x, y

    def load_all(self):
        """load vocaburary and data"""
        if os.path.isfile(self.vocab_path):
            self.vocab_processor = self.load_vocab()
        else:
            self.vocab_processor = self.train_vocab()
        if self.data_path:
            self.x, self.y = self.load_data(self.need_shuffle)
        print("Max document length: {}".format(self.max_doc))

    def batch_data(self):
        total_batch = int((len(self.x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield self.x[current:current+self.batch_size+1], \
                        self.y[current:current+self.batch_size+1]

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

    def load_vocab(self):
        return learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)

    def train_vocab(self):
        pass
