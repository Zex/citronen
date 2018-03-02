# Stack Exchange
# Dataset: https://archive.org/download/stackexchange
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
from lxml import etree
import pickle
import numpy as np

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


def to_pickle(obj, path):
    with open(path, 'wb') as fd:
        pickle.dump(obj, fd)


def from_pickle(path):
    with open(path, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


class StackEx(object):

    def __init__(self):
        self.data_path = "data/ai.stackexchange.com/Posts.xml"
        self.batch_size = 32
        self.epochs = 10
        self.max_doc_len = 128
        self.bow = set()
        self.bow_path = "data/stackex/bow.data"

        self.vocab_path = "data/stackex/vocab.data"
        #self.onehot_encoder = OneHotEncoder()
        #self.onehot_encoder_path = "data/stackex/onehot_encoder.data"
        self.prepare()

    def build_vocab_processor(self):
        self.vocab_processor = tf.contrib.learn.preprocessing\
                .text.VocabularyProcessor(\
                self.max_doc_len)

        if os.path.isfile(self.vocab_path):
            self.vocab_processor.restore(self.vocab_path)
        else:
            X = list(map(lambda x: x, self.gen_data()))
            self.vocab_processor.fit(X)
            self.vocab_processor.save(self.vocab_path)

    def prepare(self):
        self.build_vocab_processor()

        output_base = os.path.dirname(self.data_path)
        if not os.path.isdir(output_base):
            os.makedirs(output_base)

        self.build_model()

    def gen_data(self):
        with open(self.data_path) as fd:
            tree = etree.XML(fd.read())
        items = tree.xpath('row')

        for i, item in enumerate(items):
            text = item.attrib.get('Body')
            text = re.sub("<.*?>", " ", text)
            text = ' '.join(text.split()).strip()
            yield text

    def build_model(self):
        pass

    def foreach_epoch(self):
        for e in range(self.epochs):
            self.foreach_step()

    def foreach_step(self):
        for X in self.gen_data():
            X = self.vocab_processor.transform(X)
            #TODO

    def build_bow(self):
        for X in self.gen_data():
            list(map(lambda w: self.bow.add(w), X.split()))

        print("total bow: {}".format(len(self.bow)))
        to_pickle(self.bow, self.bow_path)

    def encode_text(self, text):
        def get_index(c):
            try:
                l = self.bow.index(c) + 1
            except ValueError:
                l = 0
            return l

        return list(map(lambda c: get_index(c), text))

    def preprocess(self):
#        self.bow = from_pickle(self.bow_path)
#        self.bow = list(self.bow)
        pass


if __name__ == '__main__':
    stex = StackEx()
    #list(stex.gen_data())
    stex.preprocess()
