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
        self.dropout = 0.3
        self.init_step = 0
        self.height, self.width = 75, 75
        self.channel = 2
        self.batch_size = 1604

    def _build_model(self):
        with tf.device('/cpu:0'):
            self.input_x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel), name='input_x')
            self.input_y = tf.placeholder(tf.float32, (None, 1), name='input_y')
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.logits = self.get_logits()
        self.pred = tf.argmax(self.logits, 1, name='pred')
        self.loss = tf.reduce_mean(tf.nn.softmax(self.logits, 1), name='loss')

        self.global_step = tf.Variable(self.init_step, name='global_step', trainable=False)
        self.train_op = tf.train.MomentumOptimizer(self.lr, momentum=1e-9).minimize(
                self.loss, global_step=self.global_step, name='train_op')

        summary = []
        summary.append(tf.summary.scalar('loss', self.loss))
        self.summary = tf.summary.merge(summary, name='merge_summary')

    def get_logits(self):
        with tf.device('/cpu:0'):
            conv_1 = tf.layers.conv2d(self.input_x, 3, kernel_size=[3, 3], activation=tf.nn.relu, name='conv_1')
            pool_1 = tf.layers.max_pooling2d(conv_1, [5, 5], strides=(1,1))

        with tf.device('/cpu:0'):
            conv_2 = tf.layers.conv2d(pool_1, 3, kernel_size=[3, 3], activation=tf.nn.relu, name='conv_2')
            pool_2 = tf.layers.max_pooling2d(conv_2, [5, 5], strides=(1,1))

        logits = tf.layers.dense(pool_2, 2, kernel_initializer=tf.contrib.layers.xavier_initializer()) 
        return logits

    def foreach_epoch(self, sess):
        x, y = self.preprocess()
        x = x.reshape(x.shape[0], self.height, self.width, self.channel)
        print('++ [info] shape:{} dtype:{}'.format(x.shape, x.dtype))

        feed_dict = {
               self.input_x: x,
               self.dropout_keep: 1-self.dropout,
               }

        if self.mode == Mode.TRAIN:
            feed_dict.update({self.input_y: y})
            _, loss, pred, step, summ = sess.run(\
                    [self.train_op, self.loss, self.pred, self.global_step, self.summary],\
                    feed_dict=feed_dict)
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

            for e in range(self.epochs):
                self.foreach_epoch(sess)


if __name__ == '__main__':
   Tf.start()  
