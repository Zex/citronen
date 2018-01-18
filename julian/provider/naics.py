# NAICS data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import tensorflow as tf
from julian.common.utils import raise_if_not_found
from julian.provider.data_helper import persist, from_persist
from julian.provider.data_provider import DataProvider

class NaicsProvider(DataProvider):

    def __init__(self, args, need_shuffle=True):
        self.naics_codes_path = getattr(args, "naics_codes_path", "data/naics/d6table.csv")
        self.d6table = self.load_d6table()
        self.class_map = list(set(self.d6table['code']))
        self.total_class = len(self.class_map)
        super(NaicsProvider, self).__init__(args, need_shuffle)
        self.load_all()

    def load_d6table(self):
        raise_if_not_found(self.naics_codes_path)
        return pd.read_csv(self.naics_codes_path,
                header=0, delimiter="#", dtype={"code":np.str})

    def load_data(self, need_shuffle=True):
        raise_if_not_found(self.data_path)
        chunk = pd.read_csv(self.data_path, header=0, delimiter="#", dtype={'target':np.str})
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.__process_chunk(*self.__extract_xy(chunk))

    def gen_data(self, need_shuffle=True):
        raise_if_not_found(self.data_path)
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
        header = ['code', 'name']
        df = pd.DataFrame(columns=header)
        pred = np.squeeze(pred).tolist()

        if not isinstance(pred, list):
            pred = [pred]

        for p in pred:
            code, name = self.level_decode(p)
            df = df.append(pd.Series((code, name), index=header), ignore_index=True)

        if self.pred_output:
            if os.path.isfile(self.pred_output):
                df.to_csv(self.pred_output, header=True, index=False, sep='#', mode='a')
            else:
                df.to_csv(self.pred_output, header=False, index=False, sep='#')
        return df

    def __extract_xy(self, chunk):
        chunk = chunk.dropna()
        #chunk["target"] = chunk["target"].apply(lambda x: np.int64(x[:3]))
        chunk["description"] = chunk["description"].apply(lambda l: l.lower())
        return chunk["description"], chunk["target"]

    def level_decode(self, index):
        iid = self.class_map[index]
        code = self.d6table[self.d6table["code"] == iid].values
        code = np.squeeze(code).tolist()
        return iid, code[0], code[1]

    def level_encode(self):
        pass

    def train_vocab(self):
        chunk = pd.read_csv(self.data_path, header=0, delimiter="#")
        return self.train_vocab_from_data(chunk["description"])

class NaicsStreamProvider(NaicsProvider):

    def __init__(self, args, need_shuffle=True):
        super(NaicsStreamProvider, self).__init__(args)
        self.input_stream = args.input_stream

    def batch_data(self):
        for chunk in self.input_stream:
            if isinstance(chunk, tuple): # (X, y)
                yield [self._process_chunk(chunk[0], chunk[1])]
            else:
                x = list(self.vocab_processor.transform(chunk))
                y = np.zeros(len(x))
                yield x, y
