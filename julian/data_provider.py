# Data provider for model
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from data_helper import load_l2table, load_l1table, tokenize_text, extract_xy
from data_helper import train_vocab, load_vocab, level_decode


class DataProvider(object):
    """Help prepare data for training, validation and prediction"""
    def __init__(self, args, need_shuffle=True):
        super(DataProvider, self).__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.l1table = load_l1table()
        self.l2table = load_l2table()
#        self.global_tokens = load_global_tokens()
        self.class_map = list(set(self.l2table.values()))
        self.total_class = len(self.class_map)
        self.max_doc = args.max_doc
        self.vocab_path = args.vocab_path if args.vocab_path else os.path.join(args.model_dir,
                "vocab_{}th".format(datetime.today().timetuple().tm_yday))
#       self.hv = get_hashing_vec(self.max_doc, "english")
#       self.find_bondary()
        if os.path.isfile(self.vocab_path):
            self.vocab_processor = load_vocab(self.vocab_path)
        else:
            self.vocab_processor = train_vocab(self.data_path, self.vocab_path, self.max_doc)
        self.x, self.y = self.load_data(need_shuffle)
        print("Max document length: {}".format(self.max_doc))

    def find_bondary(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            text, _, _ = extract_xy(chunk)
            tokens = tokenize_text(text)
            self.max_doc = \
                    np.max([self.max_doc, np.max([len(t) for t in tokens])])

    def batch_data(self):
        total_batch = int((len(self.x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield self.x[current:current+self.batch_size+1], \
                        self.y[current:current+self.batch_size+1]

    def load_data(self, need_shuffle=True):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="###")
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.process_chunk(*extract_xy(chunk, l2table=self.l2table))

    def gen_data(self, need_shuffle=True):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            if need_shuffle:
                chunk = shuffle(chunk)
            yield self.process_chunk(*extract_xy(chunk, l2table=self.l2table))

    def process_chunk(self, text, label1, label):
        #x = self.hv.transform(text).toarray()#[' '.join(t) for t in tokens]).toarray()
        y = []
        def init_label(lbl):
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(lbl)] = 1.
            y.append(one)

        x = list(self.vocab_processor.transform(text))
        [init_label(lbl) for lbl in label]
        #np.vectorize(init_label)(label)
        #y = tf.one_hot(label, depth=self.total_class)
        return x, y

    def decode(self, pred):
        header = ['iid', 'l1', 'l2']
        df = pd.DataFrame(columns=header)
        pred = np.squeeze(pred)
        for p in pred:
            iid, l1, l2 = level_decode(
                p,
                l1table=self.l1table,
                l2table=self.l2table,
                class_map=self.class_map)
            df = df.append(pd.Series((iid, l1, l2), index=header), ignore_index=True)
        if os.path.isfile(self.pred_output):
            df.to_csv(self.pred_output, header=True, index=False, sep='#', mode='a')
        else:
            df.to_csv(self.pred_output, header=False, index=False, sep='#')
        return df
