# Stack Exchange
# Dataset: https://archive.org/download/stackexchange
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
import sys
import string
from lxml import etree
import pickle
import numpy as np
from datetime import datetime
import ujson
import glob

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

    def __init__(self, args):
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.x_dim = args.x_dim
        self.dropout_keep = 0.7
        self.build_model()

    def build_model(self):
        with tf.name_scope('P'):
            self.w_x = tf.Variable(tf.contrib.layers.xavier_initializer()(\
                    [self.z_dim, self.h_dim]))
            self.b_x = tf.Variable(tf.zeros([self.h_dim]), dtype=tf.float32)
            self.w_log = tf.Variable(tf.contrib.layers.xavier_initializer()(\
                    [self.h_dim, self.x_dim]))
            self.b_log = tf.Variable(tf.zeros([self.x_dim]))

    def __call__(self, z):
        h = tf.nn.sigmoid(tf.nn.xw_plus_b(z, self.w_x, self.b_x))
        #h = tf.nn.batch_normalization(X, 100.258, 100.323, 0.24, 1., 1e-10)
        logits = tf.nn.xw_plus_b(h, self.w_log, self.b_log)
        prob = tf.nn.relu(logits)
        return logits, prob

    def var_list(self):
        return [self.w_x, self.w_log, self.b_x, self.b_log]


class Config(object):
    pass

class Q:

    def __init__(self, args):
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.x_dim = args.x_dim
        self.vocab_size = args.vocab_size
        self.batch_size = args.batch_size
        self.dropout_keep = 0.7
        self.build_model()

    def build_model(self):
        with tf.name_scope('Q'):
            #x = tf.nn.batch_normalization(X, 100.258, 100.323, 0.24, 1., 1e-10)
            self.w_x =  tf.Variable(tf.contrib.layers.xavier_initializer()(\
                    [self.x_dim, self.h_dim]))
            self.b_x = tf.Variable(tf.zeros([self.h_dim]))
            self.w_mu =  tf.Variable(tf.contrib.layers.xavier_initializer()([self.h_dim, self.z_dim]))
            self.b_mu = tf.Variable(tf.zeros([self.z_dim]))
            self.w_sigma =  tf.Variable(tf.contrib.layers.xavier_initializer()([self.h_dim, self.z_dim]))
            self.b_sigma = tf.Variable(tf.zeros([self.z_dim]))
             
    def __call__(self, X):
        h = tf.nn.sigmoid(tf.nn.xw_plus_b(X, self.w_x, self.b_x))
        mu = tf.nn.xw_plus_b(h, self.w_mu, self.b_mu)
        z = tf.nn.xw_plus_b(h, self.w_sigma, self.b_sigma)
        return mu, z


    def var_list(self):
        return [self.w_x, self.b_x, self.w_mu, self.b_mu, self.w_sigma, self.b_sigma]


class StackEx(object):

    def __init__(self):
        self.max_doc_len = 128

        self.z_dim = 64
        self.x_dim = self.max_doc_len
        self.h_dim = 64

        self.init_step = 1
        self.data_path = "data/ai.stackexchange.com/Posts.xml"
        self.batch_size = 32
        self.sample_size = 10

        self.bow = set()
        self.bow_path = "data/stackex/bow.data"
        self.vocab_path = "data/stackex/vocab.data"
        now = datetime.now().strftime("%Y%m%d%H%M")
        self.sample_path = "data/stackex/samples_{}.json".format(now)
        self.summ_intv = 10000
        self.epochs = 1000000
        self.clip_norm = 0.3
        self.lr = 1e-1
        self.model_dir = "models/stackex/{}/vae".format(now)
        self.prepare()

    def train_vocab(self):
        if os.path.isfile(self.vocab_path):
            self.vocab_processor = tf.contrib.learn.preprocessing\
                    .VocabularyProcessor.restore(self.vocab_path)
        else:
            self.vocab_processor = tf.contrib.learn.preprocessing\
                .text.VocabularyProcessor(\
                self.max_doc_len, min_frequency=1)

        for path in glob.iglob('data/*.stackexchange.com/Posts.xml'):
            print(path)
            self.data_path = path
            X = list(map(lambda x: x, self.gen_data()))
            self.vocab_processor = self.vocab_processor.fit(X)
            self.vocab_processor.save(self.vocab_path)
        print("[info] vocab:{}".format(len(self.vocab_processor.vocabulary_)))


    def build_vocab_processor(self):

        if os.path.isfile(self.vocab_path):
            self.vocab_processor = tf.contrib.learn.preprocessing\
                    .VocabularyProcessor.restore(self.vocab_path)
        else:
            self.vocab_processor = tf.contrib.learn.preprocessing\
                .text.VocabularyProcessor(\
                self.max_doc_len, min_frequency=1)
            X = list(map(lambda x: x, self.gen_data()))
            self.vocab_processor.fit(X)
            list(map(lambda c: self.vocab_processor.vocabulary_._mapping.get(c),\
                    string.punctuation))
            self.vocab_processor.save(self.vocab_path)
        print("[info] vocab:{}".format(len(self.vocab_processor.vocabulary_)))

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
            text = item.attrib.get('Body', item.attrib.get('Text'))
            text = re.sub("<.*?>", " ", text)
            text = ' '.join(text.split()).strip()
            yield text

    def gen_batch(self):
        batch = []

        for sample in self.gen_data():
            if len(batch) == self.batch_size:
                np.random.shuffle(batch)
                yield np.array(batch)
                batch.clear()
            else:
                batch.append(' '.join(sample.split()))
    
    def gaussian_samples(self, mu, var):
        with tf.name_scope('gaussian_samples'):
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(var/2) * eps

    def build_model(self):
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.X = tf.placeholder(tf.float32, shape=[None, self.x_dim])

        args = Config()
        args.z_dim = self.z_dim
        args.x_dim = self.x_dim
        args.h_dim = self.h_dim
        args.batch_size = self.batch_size
        args.vocab_size = len(self.vocab_processor.vocabulary_)

        self.Q = Q(args)
        self.P = P(args)

        #self.X_norm = tf.nn.batch_normalization(self.X, 256, 223, 0., 1., 1e-8)
        self.mu, self.var = self.Q(self.X)
        self.z_samples = self.gaussian_samples(self.mu, self.var)
        _, self.logits = self.P(self.z_samples)
        self.samples, _ = self.P(self.z)
        
        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(2 * self.var) - 1.+ self.mu**2, 1)
        self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.logits, labels=self.X), 1)
        self.vae_loss = tf.reduce_mean(self.recon_loss+self.kl_loss)
        
        self.global_step = tf.Variable(self.init_step)

        params = self.P.var_list()+self.Q.var_list()
        grads, global_norm = tf.clip_by_global_norm(\
                tf.gradients(self.vae_loss, params),
                #tf.trainable_variables(),
                self.clip_norm)

        """
        self.vae_train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(\
                self.vae_loss, \
                global_step=self.global_step,\
                var_list=self.P.var_list()+self.Q.var_list())
        """
        self.vae_train_op = tf.train.GradientDescentOptimizer(self.lr)\
                .apply_gradients(zip(grads, params), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    def foreach_epoch(self, sess):
        for e in range(self.epochs):
            self.foreach_step(sess)

    def foreach_step(self, sess):
        for X in self.gen_batch():
            X = list(self.vocab_processor.transform(X))
            
            if not X:
                continue

            #X = np.array(X).astype(np.float32)
            z_data = np.random.randn(self.sample_size, self.z_dim)

            _, loss, kl, recon, step = sess.run([self.vae_train_op, \
                        self.vae_loss, self.kl_loss, self.recon_loss, \
                        self.global_step],\
                feed_dict={
                    self.X: X,
                    self.z: z_data,
                    })
            
            if str(loss) == str(np.nan):
                print('[step/{}] early stop'.format(step))
                sys.exit()
            
            kl, recon = np.mean(kl), np.mean(recon)
            if step % self.summ_intv == 0:
                print('[step/{}] {} loss:{:.4} kl:{:.4} recon:{:.4}'.format(\
                        step, datetime.now(), loss, kl, recon))
                samples = sess.run(self.samples, feed_dict={self.z: z_data})
                samples = samples.astype(np.int)

                try:
                    docs = list(self.vocab_processor.reverse(samples))
                except IndexError:
                    docs = self.decode(samples)

                meta = {
                    'step': int(step),
                    'model_dir': self.model_dir,
                    'lr': float('{:.4}'.format(self.lr)),
                    'loss': float('{:.4}'.format(loss)),
                    'kl_loss': float('{:.4}'.format(kl)),
                    'recon_loss': float('{:.4}'.format(recon)),
                    'sample': docs
                    }
                self.to_json(meta, self.sample_path)
                self.saver.save(sess, self.model_dir, global_step=step)

    def decode(self, samples):
        ret = []
        for sample in np.abs(samples):
            buf = list(map(lambda a: self.vocab_processor.vocabulary_._reverse_mapping[a] \
                    if a < len(self.vocab_processor.vocabulary_) \
                else '<{}>'.format(a), sample))
            ret.append(' '.join(buf))
        return ret

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
    stex.train()
    #stex.train_vocab()
    
