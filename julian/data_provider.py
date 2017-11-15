# Data provider for model
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow.contrib import learn
import tensorflow as tf
from data_helper import persist, from_persist


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


    def load_all(self):
        """load vocaburary and data"""
        if os.path.isfile(self.vocab_path):
            self.vocab_processor = self.load_vocab()
        else:
            self.vocab_processor = self.train_vocab()
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

class SpringerProvider(DataProvider):

    def __init__(self, args, need_shuffle=True):
        self.l1_table_path = "../data/springer/l1_table.pickle"
        self.l2_table_path = "../data/springer/l2_table.pickle"
        self.load_table()
        self.class_map = list(set(self.l2table.values()))
        self.total_class = len(self.class_map)
        super(SpringerProvider, self).__init__(args, need_shuffle)
        self.load_all()

    def load_data(self, need_shuffle=True):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.__process_chunk(*self.__extract_xy(chunk))

    def gen_data(self, need_shuffle=True):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="#", chunksize=self.batch_size)
        for chunk in reader:
            if need_shuffle:
                chunk = shuffle(chunk)
            yield self.__process_chunk(*self.__extract_xy(chunk))

    def __process_chunk(self, text, label1, label2):
        y = []
        def init_label(lbl):
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(lbl)] = 1.
            y.append(one)

        x = list(self.vocab_processor.transform(text))
        list(map(init_label, label2))
        return x, y

    def __extract_xy(self, chunk):
        chunk = chunk.dropna()
    
        if self.l1table:
            chunk = chunk.replace({"cate":self.l1table})
        if self.l2table:
            chunk = chunk.replace({"subcate":self.l2table})
    
        text = chunk["desc"]
        label1 = chunk["cate"]
        label2 = chunk["subcate"]
        return text, label1, label2

    def decode(self, pred):
        header = ['iid', 'l1', 'l2']
        df = pd.DataFrame(columns=header)
        pred = np.squeeze(pred)
        for p in pred:
            iid, l1, l2 = self.level_decode(p)
            df = df.append(pd.Series((iid, l1, l2), index=header), ignore_index=True)
        if os.path.isfile(self.pred_output):
            df.to_csv(self.pred_output, header=True, index=False, sep='#', mode='a')
        else:
            df.to_csv(self.pred_output, header=False, index=False, sep='#')
        return df

    def level_decode(self, index):
        """Reversed index L1/L2 from class map"""
        iid, l1name, l2name = None, None, None
        if self.l2table:
            if not self.class_map:
                self.class_map = list(set(self.l2table.values()))
            iid = self.class_map[index]
            l2name = dict(map(reversed, self.l2table.items())).get(iid)
            if self.l1table:
                l1name = dict(map(reversed, self.l1table.items())).get(iid//0x1000*0x1000)
        return iid, l1name, l2name

    def level_encode(self):
        """Encode L1/L2"""
        if not os.path.isfile(self.l1_table_path):
            self.__encode_l1()
        if not os.path.isfile(self.l2_table_path):
            self.__encode_l2()

    def load_table(self):
        if os.path.isfile(self.l1_table_path):
            self.l1table = from_persist(self.l1_table_path)
        else:
            self.__encode_l1()

        if os.path.isfile(self.l1_table_path):
            self.l2table = from_persist(self.l2_table_path)
        else:
            self.__encode_l2()

    def __encode_l1(self):
        def assign_l1(cate):
            if cate not in self.l1_table:
                self.l1_table.update({cate:0x1000 if not self.l1_table \
                        else max(self.l1_table.values()) + 0x1000})
    
        reader = pd.read_csv(self.data_path, engine='python', header=0, chunksize=100, delimiter="#")
        [chunk["cate"].apply(assign_l1) for chunk in reader]
        persist(self.l1_table, self.l1_table_path)
    
    def __encode_l2(self):
        mem = {}
        def assign_l2(cate, subcate):
            if subcate not in self.l2_table:
                self.l2_table.update({subcate:self.l1_table[cate]+1 if cate not in mem else mem[cate]+1})
                mem.update({cate:self.l2_table[subcate]})
    
        reader = pd.read_csv(self.data_path, engine='python', header=0, chunksize=100, delimiter="#")
        [chunk.apply(lambda x: assign_l2(x["cate"], x["subcate"]), axis=1) \
            for chunk in reader]
        persist(self.l2_table, self.l2_table_path)

    def train_vocab(self):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        return self.train_vocab_from_data(chunk["desc"])

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
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.__process_chunk(*self.__extract_xy(chunk))

    def gen_data(self, need_shuffle=True):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
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
        code = self.d3table[d3table["code"] == iid].values
        return iid, code

    def level_encode(self):
        pass

    def train_vocab(self):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        return self.train_vocab_from_data(chunk["desc"])
