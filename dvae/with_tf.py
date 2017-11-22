#!/usr/bin/env python3
# Denoise VAE
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle


def xavier_init(size):
    in_dim, out_dim = size[0], size[1]
    var = np.sqrt(6/(in_dim+out_dim))
    return tf.random_normal(shape=size, stddev=var)
    #return tf.cast(tf.Variable(np.random.rand(*size)*var), tf.float32)

def random_sample(mu, log):
    return mu + tf.exp(log/2) * tf.random_normal(shape=tf.shape(mu), dtype=tf.float32)

class Base(object):
    def __init__(self, args):
        self.X_dim = args.max_doc
        self.h_dim = args.hidden_size
        self.z_dim = args.max_doc

class Enc(Base):
    def __init__(self, args):
        super(Enc, self).__init__(args)
        self.lr = args.lr
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self.w1 = tf.Variable(xavier_init([self.X_dim, self.h_dim]))
        self.b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.w_mu = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        self.b_mu = tf.Variable(tf.zeros(shape=[self.z_dim]))

        self.w_sigma = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        self.b_sigma = tf.Variable(tf.zeros(shape=[self.z_dim]))

    def __call__(self, x):
        h = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        z_mu = tf.matmul(h, self.w_mu) +  self.b_mu
        z_log = tf.matmul(h, self.w_sigma) + self.b_sigma
        return z_mu, z_log


class Dec(Base):
    def __init__(self, args):
        super(Dec, self).__init__(args)
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])

        self.w1 = tf.Variable(xavier_init([self.X_dim, self.h_dim]))
        self.b1 = tf.Variable(tf.zeros(shape=[self.h_dim]))

        self.w2 = tf.Variable(xavier_init([self.h_dim, self.z_dim]))
        self.b2 = tf.Variable(tf.zeros(shape=[self.z_dim]))

    def __call__(self, z):
        h = tf.nn.relu(tf.matmul(z, self.w1) + self.b1)
        logits = tf.matmul(h, self.w2) +  self.b2
        pred = tf.nn.sigmoid(logits)
        return logits, pred


class VAE(object):
    def __init__(self, args):
        self.enc = Enc(args)
        self.dec = Dec(args)

        self.data_path = args.data_path
        self.verbose = args.verbose
        self.model_dir = args.model_dir
        self.log_path = os.path.join(
                self.model_dir, "log_{}{}th".format(
                    datetime.today().year,
                    datetime.today().timetuple().tm_yday))

        self.eps = 1e-12
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.init_step = args.init_step
        self.summ_intv = args.summ_intv
        self.global_step = tf.Variable(self.init_step, trainable=False)
        self.lr = args.lr
        self.max_doc = args.max_doc
        self.vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(self.max_doc)
        self._build_model(args)

    def _build_model(self, args):
        noise_x = tf.clip_by_value(
            self.enc.X + tf.constant(
                args.noise_factor, dtype=tf.float32) * tf.random_normal(
                    tf.shape(self.enc.X), dtype=tf.float32),
                0., 1.)
        self.z_mu, self.z_log = self.enc(noise_x)
        self.z_sample = random_sample(self.z_mu, self.z_log)
        _, self.logits = self.dec(self.z_sample)
        self.sample_X, _ = self.dec(self.enc.z)

        self.recon_loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=self.enc.X), 1)
        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_log) + self.z_mu**2 - 1. - self.z_log, 1)
        self.loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss,
                global_step=self.global_step)

#        for k, v in params:
#            tf.summary.histogram(v.name, k)
#            tf.summary.scalar(v.name, k)

        tf.summary.scalar("loss", self.loss)
#        tf.summary.histogram("x", self.enc.X)
#        tf.summary.histogram("sample", self.sample_X)

        self.summary = tf.summary.merge_all()

    def load_data(self):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        chunk = shuffle(chunk)
        return np.array(list(
            self.vocab_processor.fit_transform(
                chunk["full_name"])), dtype=np.float32), None

    def batch_data(self):
        x, _ = self.load_data()
        total_batch = int((len(x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield x[current:current+self.batch_size+1]

    def foreach_train(self):
        with tf.Session() as sess:
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            for i in range(self.epochs):
                self.foreach_epoch(sess)
                self.saver.save(sess, self.model_dir,
                        global_step=tf.train.global_step(sess, self.global_step))

    def foreach_epoch(self, sess):
        for X  in self.batch_data():
            _, step, loss, summ = sess.run(
                    [self.train_op, self.global_step, \
                        self.loss, self.summary],
                    feed_dict={self.enc.X: X})
            if step % self.summ_intv == 0:
                self.summary_writer.add_summary(summ, step)
                sample = sess.run(self.sample_X,
                    feed_dict={self.enc.z: np.random.randn(
                        self.batch_size, self.enc.z_dim)}
                    )
                sample_text = self.vocab_processor.reverse(sample.astype(np.int32))
                print("[{}] loss:{} z:{}".format(
                    step, loss, sample), flush=True)
                with open("sample_code", 'a') as fd:
                    fd.write(("="*10+"{}"+"="*10+'\n').format(step))
                    [fd.write(str(text)) for text in sample]
                with open("sample_text", 'a') as fd:
                    fd.write(("="*10+"{}"+"="*10+'\n').format(step))
                    [fd.write(text) for text in list(sample_text)]

    def __call__(self):
        self.foreach_train()


def init():
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--data_path', default="../data/dvae/address.csv", type=str, help='Path to input data')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--summ_intv', default=100, type=int, help="Summary each several steps")
    parser.add_argument('--init_step', default=0, type=int, help="Initial training step")
    parser.add_argument('--epochs', default=10000, type=int, help="Total epochs to train")
    parser.add_argument('--noise_factor', default=0.30, type=float, help="Noise factor")
    parser.add_argument('--max_doc', default=512, type=int, help="Maximum document length")
    parser.add_argument('--hidden_size', default=128, type=int, help="Hidden layer size")
    parser.add_argument('--model_dir', default="../models/dvae", type=str, help="Path to model and check point")
    parser.add_argument('--verbose', default=False, action='store_true', help="Print verbose")

    return parser.parse_args()

def start():
    args = init()
    vae = VAE(args)
    vae()

if __name__ == '__main__':
    start()
