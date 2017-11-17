# NAICS data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import tensorflow as tf
from julian.provider.data_helper import persist, from_persist
from julian.provider.data_provider import DataProvider

class NaicsProvider(DataProvider):

    def __init__(self, args, need_shuffle=True):
        self.naics_codes_path = "../data/naics/codes_3digits.csv"
        self.d3_table_path = "../data/naics/d3_table.pickle"
        self.d3table = self.load_d3table()
        self.class_map = list(set(self.d3table['code']))
        self.total_class = len(self.class_map)
        super(NaicsProvider, self).__init__(args, need_shuffle)
        self.load_all()

    def load_d3table(self):
        return pd.read_csv(self.naics_codes_path, engine='python',
                header=0, delimiter="#", dtype={"code":np.int})
    
    def load_data(self, need_shuffle=True):
        chunk = pd.read_csv(self.data_path, header=0, delimiter="#")
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.__process_chunk(*self.__extract_xy(chunk))

    def gen_data(self, need_shuffle=True):
        reader = pd.read_csv(self.data_path, header=0,
            delimiter="#", chunksize=self.batch_size)
        for chunk in reader:
            if need_shuffle:
                chunk = shuffle(chunk)
            yield self.__process_chunk(*self.__extract_xy(chunk))

    def __process_chunk(self, text, label):
        y = []
        def init_label(lbl):
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(lbl)] = 1.
            y.append(one)

        x = list(self.vocab_processor.transform(text))
        list(map(init_label, label))
        return x, y

    def decode(self, pred):
        header = ['iid', 'code']
        df = pd.DataFrame(columns=header)
        pred = np.squeeze(pred)

        for p in pred:
            iid, code = self.level_decode(p)
            df = df.append(pd.Series((iid, code), index=header), ignore_index=True)

        if self.pred_otuput:
            if os.path.isfile(self.pred_output):
                df.to_csv(self.pred_output, header=True, index=False, sep='#', mode='a')
            else:
                df.to_csv(self.pred_output, header=False, index=False, sep='#')
        return df

    def __extract_xy(self, chunk):
        chunk = chunk.dropna()
        chunk["code"] = chunk["code"].apply(lambda x: np.int64(x[:3]))
        return chunk["desc"], chunk["code"]

    def level_decode(self, index):
        iid = self.class_map[index]
        code = self.d3table[self.d3table["code"] == iid].values
        return iid, code

    def level_encode(self):
        pass

    def train_vocab(self):
        chunk = pd.read_csv(self.data_path, header=0, delimiter="#")
        return self.train_vocab_from_data(chunk["desc"])
