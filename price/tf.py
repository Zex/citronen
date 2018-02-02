# TF 
#
import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from sklearn.utils import shuffle
import pickle
from price.provider import *


class Price(object):

    def __init__(self, args):
        super(Price, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.init_step = args.init_step
        self.dropout_rate = args.dropout_rate
        self.summ_intv = args.summ_intv
        self.model_dir = args.model_dir
        self.log_path = os.path.join(self.model_dir, 'cnn')
        self.total_feat = 942
        self.channel = 1
        self.cfg = Config(args)

    def _build_model(self):
        with tf.device('/cpu:0'):
            self.input_x = tf.placeholder(tf.float32, (None, self.total_feat, self.channel), name='input_x')
            self.input_y = tf.placeholder(tf.float32, (None, 1), name='input_y')
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.layers = []
        total_layers = 3
        n_filters = 5

        with tf.device('/cpu:0'):
            for i in range(total_layers):
                if not self.layers:
                    input_x = self.input_x
                else:
                    input_x = self.layers[-1]

                conv = tf.layers.conv1d(input_x, \
                        n_filters, kernel_size=5, name='conv_{}'.format(i),\
                        activation=tf.nn.relu, \
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),\
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.3))
                pool = tf.layers.max_pooling1d(conv, [3], [1], name='pool_{}'.format(i))
                self.layers.append(pool)

        print(self.layers)

        #flat = tf.reshape(self.layers[-1], [-1, 912*3])
        flat = tf.reshape(self.layers[-1], [-1, 924*n_filters])
        hidden = tf.nn.dropout(flat, self.dropout_keep)

        with tf.device('/cpu:0'):
            self.logits = tf.layers.dense(hidden, 1, use_bias=True,\
                activation=tf.nn.relu,\
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='logits')

        #print('logits', self.logits)
        self.loss = tf.reduce_mean(tf.pow(tf.log(self.logits+1)-tf.log(self.input_y+1), 2), name='loss')
        #self.loss = tf.losses.log_loss(self.input_y, self.logits)
        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(\
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
            sess.run(tf.global_variables_initializer())
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(tf.global_variables())

            for e in range(self.epochs):
                self.foreach_epoch(sess, e)
     
    def foreach_epoch(self, sess, e):
        for X, y in preprocess(self.cfg):
            feed_dict = {
                self.input_x: X,
                self.input_y: y,
                self.dropout_keep: 1-self.dropout_rate,
                }

            _, loss, pred, step, summ = sess.run(\
                [self.train_op, self.loss, self.logits, self.global_step, self.summary],\
                feed_dict=feed_dict)
            pred = np.squeeze(pred)

            if step % self.summ_intv == 0:
                print('++ [step-{}] loss:{} pred:{}'.format(step, loss, pred))
                self.summary_writer.add_summary(summ, step)
                self.saver.save(sess, self.model_dir+'/cnn',
                            global_step=tf.train.global_step(sess, self.global_step))
                self.inner_test(sess, step)
    

    def inner_test(self, sess, step):
        def foreach_chunk(iid, X):
            feed_dict = {
               self.input_x: X,
               self.dropout_keep: 1,
               }
    
            pred = sess.run([self.logits], feed_dict=feed_dict)    
            pred = np.squeeze(pred)

            df = pd.DataFrame({
                    'test_id': iid,
                    'price': pred,
                })

            to_csv(df, cfg.result_path)

        cfg = Config()

        cfg.data_path = "data/price/test.tsv"
        cfg.need_shuffle = False
        cfg.mode = Mode.TEST
        cfg.result_path = "data/price/pred_tf_{}_{}.csv".format(\
                    step,
                    datetime.now().strftime("%y%m%d%H%M"))
    
        gen = preprocess(cfg)
        list(map(lambda z: foreach_chunk(z[0], z[1]), gen))


def start():
    obj = Price(init())
    obj.train()

if __name__ == '__main__':
    start()
