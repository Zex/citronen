# Stack Exchange
# Dataset: https://archive.org/download/stackexchange
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
from lxml import etree
import pickle
import numpy as np
import ujson

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


def to_pickle(obj, path):
    with open(path, 'wb') as fd:
        pickle.dump(obj, fd)


def from_pickle(path):
    with open(path, 'rb') as fd:
        ret = pickle.load(fd)
    return ret


class P:

    def __init__(self, z_dim, h_dim, x_dim):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim

        with tf.name_scope('P'):
            self.w1 = tf.Variable(tf.contrib.layers.xavier_initializer()([self.z_dim, self.h_dim]))
            self.b1 = tf.Variable(tf.zeros([self.h_dim,]))
            self.w2 = tf.Variable(tf.contrib.layers.xavier_initializer()([self.h_dim, self.x_dim]))
            self.b2 = tf.Variable(tf.zeros([self.h_dim,]))

    def __call__(self, z):
        with tf.name_scope('P'):
            h = tf.nn.relu(tf.nn.xw_plus_b(z, self.w1, self.b1, name='l_1'))
            logits = tf.nn.xw_plus_b(h, self.w2, self.b2, name='l_2')
            prob = tf.nn.sigmoid(logits)
        return prob, logits

    def var_list(self):
        return [self.w1, self.b1, self.w2, self.b2]


class Q:

    def __init__(self, z_dim, h_dim, x_dim):
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim

        with tf.name_scope('Q'):
            self.w1 = tf.Variable(tf.contrib.layers.xavier_initializer()([self.x_dim, self.h_dim]))
            self.b1 = tf.Variable(tf.zeros([self.h_dim,]))
            self.w2 = tf.Variable(tf.contrib.layers.xavier_initializer()([self.h_dim, self.z_dim]))
            self.b2 = tf.Variable(tf.zeros([self.z_dim,]))

    def __call__(self, X):
        with tf.name_scope('Q'):
            h = tf.nn.relu(tf.nn.xw_plus_b(X, self.w1, self.b1, name='l_1'))
            z = tf.nn.relu(tf.nn.xw_plus_b(h, self.w2, self.b2, name='l_1'))
        return z

    def var_list(self):
        return [self.w1, self.b1, self.w2, self.b2]


class StackEx(object):

    def __init__(self):
        self.max_doc_len = 128

        self.z_dim = 10
        self.x_dim = self.max_doc_len
        self.h_dim = 128

        self.init_step = 0
        self.data_path = "data/ai.stackexchange.com/Posts.xml"
        self.batch_size = 32

        self.bow = set()
        self.bow_path = "data/stackex/bow.data"
        self.vocab_path = "data/stackex/vocab.data"
        self.summ_intv = 1000
        self.epochs = 1000000
        self.sample_path = "data/stackex/samples.json"
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
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_dim])

        self.Q = Q(self.z_dim, self.h_dim, self.x_dim)
        self.P = P(self.z_dim, self.h_dim, self.x_dim)

        self.z_samples = self.Q(self.X)
        _, self.logits = self.P(self.z_samples)
        self.samples, _ = self.P(self.z)

        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.logits))
        self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.logits, labels=self.X), 1)
        self.vae_loss = tf.reduce_mean(self.kl_loss + self.recon_loss)

        self.global_step = tf.Variable(self.init_step)
        self.vae_train_op = tf.train.AdamOptimizer().minimize(self.vae_loss, global_step=self.global_step)

    def foreach_epoch(self, sess):
        for e in range(self.epochs):
            self.foreach_step(sess)

    def foreach_step(self, sess):
        for X in self.gen_data():
            X = list(self.vocab_processor.transform(X))
            if not X:
                continue
            _, loss, step = sess.run([self.vae_train_op, self.vae_loss, self.global_step], \
                feed_dict={self.X: X})

            if step % self.summ_intv == 0:
                print('[step/{}] loss:{:.4}'.format(step, loss))
                samples = sess.run(self.samples, feed_dict={self.z: np.random.randn(50, self.z_dim)})
                docs = list(self.vocab_processor.reverse(samples.astype(np.int)))
                self.to_json({'sample': docs}, self.sample_path)

    def to_json(self, obj, output_path):
        with open(output_path, 'a') as fd:
            fd.write(ujson.dumps(obj)+'\n')

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(self.epochs):
                self.foreach_epoch(sess)

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
    #stex.preprocess()
    stex.train()
