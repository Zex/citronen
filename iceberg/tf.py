# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import sys
import os
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from iceberg.iceberg import Iceberg, Mode


class Tf(Iceberg):

    def __init__(self, args):
        super(Tf, self).__init__(args)
        self.dropout = 0.1
        self.init_step = 0
        self.height, self.width = 75, 75
        self.channel = 2
        self.summ_intv = 100

    def _build_model(self):
        with tf.device('/cpu:0'):
            self.input_x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel), name='input_x')
            self.input_y = tf.placeholder(tf.float32, (None, 1), name='input_y')
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.logits = self.get_logits()
        self.pred = tf.argmax(self.logits, -1, name='pred')
        #self.loss = tf.reduce_mean(tf.nn.softmax(self.logits, 1), name='loss')
        self.loss = tf.losses.log_loss(self.input_y, self.logits)

        self.global_step = tf.Variable(self.init_step, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, global_step=self.global_step, name='train_op')
        #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
        #        self.loss, global_step=self.global_step, name='train_op')

        summary = []
        summary.append(tf.summary.scalar('loss', self.loss))
        summary.append(tf.summary.histogram('logits', self.logits))
        self.summary = tf.summary.merge(summary, name='merge_summary')

    def get_logits(self):
        with tf.device('/cpu:0'):
            conv_1 = tf.layers.conv2d(self.input_x, 5, kernel_size=[5, 5], activation=tf.sigmoid, \
                name='conv_1', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_1 = tf.layers.max_pooling2d(conv_1, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_2 = tf.layers.conv2d(self.input_x, 3, kernel_size=[3, 3], activation=tf.sigmoid, name='conv_2',\
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_2 = tf.layers.max_pooling2d(conv_2, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_3 = tf.layers.conv2d(pool_1, 5, kernel_size=[5, 5], activation=tf.sigmoid,\
                name='conv_3', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_3 = tf.layers.max_pooling2d(conv_3, [5, 5], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_4 = tf.layers.conv2d(pool_2, 3, kernel_size=[3, 3], activation=tf.sigmoid, name='conv_4',\
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_4 = tf.layers.max_pooling2d(conv_4, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_5 = tf.layers.conv2d(pool_3, 3, kernel_size=[3, 3], activation=tf.sigmoid,\
                name='conv_5', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_5 = tf.layers.max_pooling2d(conv_5, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_6 = tf.layers.conv2d(pool_4, 3, kernel_size=[3, 3], activation=tf.sigmoid,\
                name='conv_6', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_6 = tf.layers.max_pooling2d(conv_6, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_7 = tf.layers.conv2d(pool_5, 3, kernel_size=[3, 3], activation=tf.sigmoid, name='conv_7')
            pool_7 = tf.layers.max_pooling2d(conv_7, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_8 = tf.layers.conv2d(pool_6, 3, kernel_size=[3, 3], activation=tf.sigmoid, name='conv_8',\
                kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_8 = tf.layers.max_pooling2d(conv_8, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_9 = tf.layers.conv2d(pool_7, 5, kernel_size=[5, 5], activation=tf.sigmoid, name='conv_9')
            pool_9 = tf.layers.max_pooling2d(conv_9, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_10 = tf.layers.conv2d(pool_9, 3, kernel_size=[5, 5], activation=tf.sigmoid,\
                name='conv_10', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_10 = tf.layers.max_pooling2d(conv_10, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            conv_11 = tf.layers.conv2d(pool_10, 3, kernel_size=[5, 5], activation=tf.sigmoid,\
                name='conv_11', kernel_initializer=tf.contrib.layers.xavier_initializer())
            pool_11 = tf.layers.max_pooling2d(conv_11, [3, 3], strides=(1, 1))

        with tf.device('/cpu:0'):
            #hidden = tf.concat([pool_7, pool_10], 3)
            #hidden = tf.reshape(pool_7, [-1, 53*53*3], name='hidden')
            #hidden = tf.reshape(pool_5, [-1, 57*57*3], name='hidden')
            hidden = tf.reshape(pool_11, [-1, 35*35*3], name='hidden')
            logits = tf.layers.dense(hidden, 1, use_bias=True, activation=tf.sigmoid, name='logits',\
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        return logits

    def batch_data(self):
        X, y = self.X, self.y
        X = X.reshape(X.shape[0], self.height, self.width, self.channel)
        begin = 0
        
        while begin < len(X):
            end = min(begin+self.batch_size, len(X))
            Xtrain, ytrain = X[begin:end], y[begin:end]
            yield Xtrain, ytrain
            begin = end

    def foreach_epoch(self, sess):
        for Xtrain, ytrain in self.batch_data():
            self.foreach_step(sess, Xtrain, ytrain)

    def foreach_step(self, sess, X, y):
        #print('++ [info] X: shape:{} dtype:{}'.format(X.shape, X.dtype), flush=True)
        #print('++ [info] y: shape:{} dtype:{}'.format(y.shape, y.dtype), flush=True)

        feed_dict = {
               self.input_x: X,
               self.dropout_keep: 1-self.dropout,
               }

        if self.mode == Mode.TRAIN:
            feed_dict.update({self.input_y: y})
            _, loss, pred, step, summ = sess.run(\
                    [self.train_op, self.loss, self.logits, self.global_step, self.summary],\
                    feed_dict=feed_dict)
            if step % self.summ_intv == 0:
                self.summary_writer.add_summary(summ, step)
                self.saver.save(sess, self.model_dir+'/cnn',
                            global_step=tf.train.global_step(sess, self.global_step))
                pred = np.squeeze(pred)
                print('++ [step:{}] loss:{} pred:{}'.format(step, loss, pred), flush=True)

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"

        self._build_model()

        with tf.Session() as sess:
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            self.X, self.y = self.preprocess()

            for e in range(self.epochs):
                self.foreach_epoch(sess)


if __name__ == '__main__':
   Tf.start()  
