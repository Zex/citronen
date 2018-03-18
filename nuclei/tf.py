# Nuclei TF
# Author: Zex Li <top_zlynch@yahoo.com>
import numpy as np
import tensorflow as tf
from nuclei.provider import *


class TF(object):

    def __init__(self, args):
        super(TF, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.init_step = args.init_step
        self.dropout_rate = args.dropout_rate
        self.summ_intv = args.summ_intv
        self.model_dir = args.model_dir
        self.log_path = os.path.join(self.model_dir, 'cnn')

        self.prov = Provider()
        self.height, self.width = self.prov.height, self.prov.width
        self.channel = self.prov.channel
        self.device = "/cpu:0"


    def _build_model(self):
        with tf.device(self.device):
            self.input_x = tf.placeholder(tf.float32, (None, self.height, self.width, self.channel), name='input_x')
            self.input_y = tf.placeholder(tf.float32, (None, self.height, self.width), name='input_y')
            self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')
            self.total_nuclei = tf.placeholder(tf.int32, (None, 1), name='total_nuclei')

        self.layers = []
        total_layers = 3
        n_filters = 50

        with tf.device(self.device):
            for i in range(total_layers):
                if not self.layers:
                    input_x = self.input_x
                else:
                    input_x = self.layers[-1]

                conv = tf.layers.conv2d(input_x, \
                        n_filters, kernel_size=[3, 3], name='conv_{}'.format(i),\
                        activation=tf.nn.relu, \
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       # kernel_regularizer=tf.contrib.layers.l2_regularizer(0.3)
                        )
                pool = tf.layers.max_pooling2d(conv, [3, 3], [3, 3], name='pool_{}'.format(i))
                self.layers.append(conv)#pool)
        [print(l) for l in self.layers]
        self.loss = tf.reduce_mean(self.layers[-1])#tf.metrics.mean_iou(self.input_x, self.input_x, self.total_nuclei, name='loss')
        
        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss, global_step=self.global_step, name='train_op')

        summary = []
        summary.append(tf.summary.scalar('loss', self.loss))
        #summary.append(tf.summary.histogram('logits', self.logits))
        summary.append(tf.summary.image('input_x', self.input_x))
        summary.extend([tf.summary.histogram('pool_{}'.format(i), pool) for i, pool in enumerate(self.layers)])
        #summary.extend([tf.summary.image('pool_{}'.format(i), pool) for i, pool in enumerate(self.layers)])
        summary.extend([tf.summary.image(var.name, var) for var in tf.global_variables() if \
                var.name.startswith('conv_') and var.name.endswith('kernel:0')])
        #summary.extend([tf.summary.image('/kernels', grid, max_outputs=1) 
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
        for X, y, total_nuclei in self.prov.gen_data():
            feed_dict = {
                self.input_x: X,
                self.input_y: y,
                self.dropout_keep: 1-self.dropout_rate,
                self.total_nuclei: total_nuclei,
                }

            _, loss, step, summ = sess.run(\
                [self.train_op, self.loss, self.global_step, self.summary],\
                feed_dict=feed_dict)

            if step % self.summ_intv == 0:
                print('++ [step-{}] loss:{}'.format(step, loss), flush=True)
                self.summary_writer.add_summary(summ, step)
                self.saver.save(sess, self.model_dir+'/cnn',
                            global_step=tf.train.global_step(sess, self.global_step))
                self.inner_test(sess, step)
        
                
    def inner_test(self, sess, step):
        pass

def start():
    obj = TF(init())
    obj.train()


if __name__ == "__main__":
    start()
