#
#
import os
import sys
import glob
import pandas as pd
import numpy as np
#import seaborn as sns
#from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle


def preprocess():
    path = 'data/price/train.tsv'
    batch_size = 1000
    gen = pd.read_csv(path, delimiter='\t', chunksize=batch_size)

    for df in gen:
        df = df.drop_duplicates()
        cate = encode_cate(df)
        name = encode_name(df)
        desc = encode_desc(df)

        X = np.concatenate((cate, name, desc), 1)
        X = np.concatenate((X, df['shipping'].values.reshape(X.shape[0], 1)), 1).astype(np.float)
        target = df['price'].astype(np.float)

        yield X, target

def encode_cate(df):
    path = 'data/price/cate.pickle'
    le, cate = load_or_fit(df, path, 'category_name')
    return cate
        
def encode_name(df):
    path = 'data/price/name.pickle'
    le, name = load_or_fit(df, path, 'name')
    return name

def encode_desc(df):
    path = 'data/price/desc.pickle'
    le, desc = load_or_fit(df, path, 'item_description')
    return desc


def load_or_fit(df, path, field):
    df[field] = df[field].fillna('')

    if not os.path.isfile(path):
        le = TfidfVectorizer()
        le = le.fit(df[field])

        with open(path, 'wb') as fd:
            pickle.dump(le, fd)
    else:
        with open(path, 'rb') as fd:
            le = pickle.load(fd)
    
    ret = le.transform(df[field]).toarray()
    return le, ret#.reshape(ret.shape[0], ret.shape[1])


class Price(object):

    def __init__(self):
        super(Price, self).__init__()
        self.epochs = 1000000
        self.lr = 1e-3
        self.init_step = 0
        self.dropout_rate = 0.4
        self.summ_intv = 30
        self.model_dir = "models/price"
        self.total_feat = 268801
        self.channel = 1

    def _build_model(self):
        with tf.device('/cpu:0'):
            self.input_x = tf.placeholder(tf.float32, (None, self.total_feat, self.channel), name='input_x')
            self.input_y = tf.placeholder(tf.float32, (None, 1), name='input_y')
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')


        self.layers = []
        total_layers = 5

        for i in range(total_layers):
            with tf.device('/cpu:0'):
                if not self.layers:
                    conv = tf.layers.conv1d(self.input_x, 5, kernel_size=5, name='conv_{}'.format(i))
                else:
                    conv = tf.layers.conv1d(self.layers[-1], 5, kernel_size=5, name='conv_{}'.format(i))
                pool = tf.layers.max_pooling1d(conv, [5], [1], name='pool_{}'.format(i))
                self.layers.append(pool)

        print(self.layers)

        flat = tf.reshape(self.layers[-1], [-1, 50*50])
        hidden = tf.nn.dropout(flat, self.dropout_keep)

        with tf.device('/cpu:0'):
            w = tf.Variable(shape=[self.hidden.get_shape()[1], 1],\
                    initializer=tf.contrib.layers.xavier_initializer(),\
                    name='logits_w')
            b = tf.constant(0.1, name='logits_b')
            self.logits = tf.nn.xw_plus_b(hidden, w, b, name='logits')

        self.loss = tf.reduce_mean(tf.pow(tf.log(self.logits+1)-tf.log(self.input_y+1), 2), name='loss')
        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        self.train_op = tf.train.RMSPropOptimizer(self.lr, momentum=0.9).minimize(\
            self.loss, global_step=self.global_step, name='train_op')

        summary = []
        summary.append(tf.summary.scalar('loss', self.loss))
        summary.append(tf.summary.histogram('logits', self.logits))
        summary.extend([tf.summary.histogram('pool_{}'.format(i), pool) for i, pool in enumerate(self.layers)])
        self.summary = tf.summary.merge(summary, name='summary')

    def train(self):
        self._build_model()
        #self.estimator.train(input_fn=preprocess, steps=100)
        with tf.Session() as sess:
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            for e in range(self.epochs):
                self.foreach_epoch(sess, e)
     
    def foreach_epoch(self, sess, e):
        for X, y in preprocess():
            feed_dict = {
                self.input_x: X,
                self.input_y: y,
                self.dropout_keep: 1-self.dropout_rate,
                }

            _, loss, pred, step, summ = sess.run(\
                [self.train_op, self.loss, self.logits, self.global_step, self.summary],\
                feed_dict=feed_dict)

            if step % self.summ_intv == 0:
                print('++ [step-{}] loss:{} pred:{}'.format(step, loss, pred))
                self.summary_writer.add_summary(summ, step)
                self.saver.save(sess, self.model_dir+'/cnn',
                            global_step=tf.train.global_step(sess, self.global_step))
    

def start():
    #for x, y in preprocess():
    #    print(x.shape)
    obj = Price()
    obj.train()

if __name__ == '__main__':
    start()
