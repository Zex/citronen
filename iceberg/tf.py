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
        self.dropout = 0.7
        self.init_step = 0

    def _build_model(self):
        self.input_x = tf.placeholder(tf.int32, (None, 75, 75, 1), name='input_x')
        self.input_y = tf.placeholder(tf.float32, (None, 1), name='input_y')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.logits = self.get_logits()
        self.pred = tf.argmax(self.logits, 1, name='pred')
        self.loss = tf.reduce_mean(tf.nn.log_softmax(self.logits, 1), name='loss')

        self.global_step = tf.Variable(self.init_step, name='global_step', trainable=False)
        self.train_op = tf.train.MomentumOptimizer(self.lr, momentum=1e-9).minimize(
                self.loss, global_step=self.global_step, name='train_op')

        summary = []
        summary.append(tf.summary.scalar('loss', self.loss))
        self.summary = tf.summary.merge(summary, name='merge_summary')

    def get_logits(self):
        conv_1 = tf.layers.conv2d(self.input_x, 3, kernel_size=[3, 3], activation=tf.nn.relu, name='conv_1')
        pool_1 = tf.layers.max_pooling2d(conv_1, [5, 5], strides=(1,1))

        conv_2 = tf.layers.conv2d(pool_1, 3, kernel_size=[3, 3], activation=tf.nn.relu, name='conv_2')
        pool_2 = tf.layers.max_pooling2d(conv_2, [5, 5], strides=(1,1))

        logits = tf.layers.dense(pool_2, 2, kernel_initializer=tf.contrib.layers.xavier_initializer()) 
        return logits

    def foreach_epoch(self, sess):
        x, y = self.preprocess()
        x = x.reshape(x.shape[0], 75, 75, 1)

        if True:
            feed_dict = {
                    self.input_x: x,
                    self.dropout_keep: self.dropout,
                    }

            if self.mode == Mode.TRAIN:
                feed_dict.update({self.input_y: y})
            _, step, loss, pred = sess.run([self.train_op, self.loss, self.pred])

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"

        self._build_model()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(self.epochs):
                self.foreach_epoch(sess)


if __name__ == '__main__':
   Tf.start()  
