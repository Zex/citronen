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

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

class Config(object):
    pass


class P:

    def __init__(self, args):
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.x_dim = args.x_dim
        self.vocab_size = args.vocab_size
        self.batch_size = args.batch_size
        self.h_dim = self.batch_size#self.vocab_size
        self.emb_dim = 100
        self.rnn_size = 512
        self.rnn_steps = self.z_dim
        self.dropout_keep = 0.7
        self.build_model()

    def build_model(self):
        with tf.name_scope('P'):
            self.z = tf.placeholder(tf.int32, shape=[None, self.z_dim], name='z')

            self.w_x = tf.Variable(tf.random_normal_initializer()(\
                    [self.z_dim, self.h_dim]))
            self.b_x = tf.Variable(tf.zeros([self.h_dim]), dtype=tf.float32)
            self.w_log = tf.Variable(tf.random_normal_initializer()(\
                    [self.h_dim, self.x_dim]))
            self.b_log = tf.Variable(tf.zeros([self.x_dim]))
            
            emb = tf.get_variable('P/emb', [self.vocab_size, self.rnn_size], \
                    tf.float32, tf.random_normal_initializer())
            emb_output = tf.nn.embedding_lookup(emb, self.z)
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, reuse=True)
            self.rnn_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)

            outputs = []
            for step in range(self.rnn_steps):
                output, self.rnn_state = self.rnn_cell(\
                        emb_output[:,step,:], self.rnn_state)
                outputs.append(output)

            self.rnn_output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
            self.w_rnn = tf.get_variable('P/w_rnn', [self.rnn_size, self.vocab_size], \
                    tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_rnn = tf.Variable(tf.zeros([self.vocab_size]))
        
            h = tf.nn.xw_plus_b(self.rnn_output, self.w_rnn, self.b_rnn)
            self.logits = tf.nn.xw_plus_b(h, self.w_log, self.b_log)
            self.prob = tf.nn.relu(self.logits)
        
            
    def __call__(self, sess, z):
        #h = tf.nn.relu(tf.nn.xw_plus_b(z, self.w_x, self.b_x))
        #logits = tf.nn.xw_plus_b(h, self.w_log, self.b_log)
        #prob = tf.nn.relu(logits)
        logits, prob = sess.run({self.logits, self.prob}, feed_dict={self.z: z})
        return logits, prob

    def var_list(self):
        return [self.w_rnn, self.b_rnn, self.w_x, self.w_log, self.b_x, self.b_log]



class Q:

    def __init__(self, args):
        self.z_dim = args.z_dim
        #self.h_dim = args.h_dim
        self.x_dim = args.x_dim
        self.vocab_size = args.vocab_size
        self.batch_size = args.batch_size
        self.h_dim = self.vocab_size
        self.emb_dim = 100
        self.rnn_size = 512
        self.rnn_steps = self.x_dim
        self.dropout_keep = 0.7
        self.build_model()

    def build_model(self):
        with tf.name_scope('Q'):
            self.X = tf.placeholder(tf.int32, shape=[None, self.x_dim], name='X')
            #x = tf.nn.batch_normalization(X, 100.258, 100.323, 0.24, 1., 1e-10)
            emb = tf.get_variable('Q/emb', [self.vocab_size, self.rnn_size], tf.float32, tf.random_normal_initializer())
            emb_output = tf.nn.embedding_lookup(emb, self.X)
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            self.rnn_state = self.rnn_cell.zero_state(self.batch_size, tf.float32)

            outputs = []
            for step in range(self.rnn_steps):
                output, self.rnn_state = self.rnn_cell(\
                        emb_output[:,step,:], self.rnn_state)
                outputs.append(output)

#            self.rnn_inputs = tf.split(axis=1, \
#                    num_or_size_splits=self.x_dim, \
#                    value=emb_output)
#
#            self.rnn_inputs_items = [tf.squeeze(x, [1]) for x in self.rnn_inputs]
#            outputs, self.final_state = tf.contrib.legacy_seq2seq.rnn_decoder(
#                    self.rnn_inputs_items, self.rnn_state,
#                    self.rnn_cell
#                    )
            self.rnn_output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
            print(self.rnn_output)
            self.conv_input = tf.reshape(self.rnn_output, [self.batch_size, *self.rnn_output.shape])
            conv = tf.nn.relu(tf.contrib.layers.conv2d(self.rnn_output, 3, [3, 3]))
            pool = tf.nn.max_pool(conv, [3, 3])

            print(pool)

            self.w_rnn = tf.get_variable('Q/w_rnn', [self.rnn_size, self.vocab_size], \
                    tf.float32, tf.contrib.layers.xavier_initializer())
            self.b_rnn = tf.Variable(tf.zeros([self.vocab_size]))
            
            self.w_mu =  tf.Variable(tf.contrib.layers.xavier_initializer()(\
                    [self.h_dim, self.z_dim]))
            self.b_mu = tf.Variable(tf.zeros([self.z_dim]))
            self.w_sigma =  tf.Variable(tf.contrib.layers.xavier_initializer()(\
                    [self.h_dim, self.z_dim]))
            self.b_sigma = tf.Variable(tf.zeros([self.z_dim]))

            h = tf.nn.xw_plus_b(self.rnn_output, self.w_rnn, self.b_rnn)
            self.mu = tf.nn.xw_plus_b(h, self.w_mu, self.b_mu)
            self.var = tf.nn.xw_plus_b(h, self.w_sigma, self.b_sigma)
            self.z_samples = self.gaussian_samples(self.mu, self.var)

    def gaussian_samples(self, mu, var):
        with tf.name_scope('gaussian_samples'):
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(var/2) * eps

    def __call__(self, sess, X):
        z_samples  = sess.run(self.z_samples, feed_dict={self.X: X})
        return z_samples

    def var_list(self):
        return [self.w_rnn, self.b_rnn, self.w_mu, self.b_mu, self.w_sigma, self.b_sigma]


class StackEx(object):

    def __init__(self):
        self.max_doc_len = 128 #256

        self.z_dim = 128
        self.x_dim = self.max_doc_len
        self.h_dim = 64

        self.init_step = 1
        self.clip_norm = 0.3
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
        self.lr = 1e-4
        self.model_dir = "models/stackex/{}/rnn".format(now)
        self.prepare()

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
        output_base = os.path.dirname(self.vocab_path)
        if not os.path.isdir(output_base):
            os.makedirs(output_base)

        self.build_vocab_processor()
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

    def gen_batch(self):
        batch = []

        for sample in self.gen_data():
            if len(batch) == self.batch_size:
                np.random.shuffle(batch)
                yield np.array(batch)
                batch.clear()
            else:
                batch.append(' '.join(sample.split()))
    

    def build_model(self):
        #self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
        #self.X = tf.placeholder(tf.int32, shape=[None, self.x_dim], name='X')
        #self.z = tf.placeholder(tf.int32, shape=[None, self.z_dim], name='z')
        #self.X = tf.placeholder(tf.int32, shape=[None, self.x_dim], name='X')

        args = Config()
        args.z_dim = self.z_dim
        args.x_dim = self.x_dim
        args.h_dim = self.h_dim
        args.batch_size = self.batch_size
        args.vocab_size = len(self.vocab_processor.vocabulary_)

        self.Q = Q(args)
        self.P = P(args)

        #self.X_norm = tf.nn.batch_normalization(self.X, 256, 223, 0., 1., 1e-8)
        #self.mu, self.var = self.Q(self.X)
        #self.z_samples = self.gaussian_samples(self.mu, self.var)
        #_, self.logits = self.P(self.z_samples)
        #self.samples, _ = self.P(self.z)
        
        #self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(2 * self.var) - 1.+ self.mu**2, 1)
        #self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(\
        #        logits=self.logits, labels=tf.cast(self.X, tf.float32)), 1)
        #self.vae_loss = tf.reduce_mean(self.recon_loss+self.kl_loss)
        self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(2 * self.Q.var) - 1.+ self.Q.mu**2, 1)
        self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.P.logits, labels=tf.cast(self.Q.X, tf.float32)), 1)
        self.vae_loss = tf.reduce_mean(self.kl_loss+self.recon_loss)
        
        self.global_step = tf.Variable(self.init_step)
        grads, global_norm = tf.clip_by_global_norm(\
                tf.gradients(self.vae_loss, self.P.var_list()+self.Q.var_list()),
                #tf.trainable_variables(),
                self.clip_norm)

        """
        self.vae_train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(\
                self.vae_loss, \
                global_step=self.global_step,\
                var_list=self.P.var_list()+self.Q.var_list())
        """
        self.vae_train_op = tf.train.GradientDescentOptimizer(self.lr)\
                .apply_gradients(zip(grads, self.P.var_list()+self.Q.var_list()))
        self.saver = tf.train.Saver(tf.global_variables())

    def foreach_epoch(self, sess):
        for e in range(self.epochs):
            self.foreach_step(sess)

    def foreach_step(self, sess):
        for X in self.gen_batch():
            X = list(self.vocab_processor.transform(X))
            
            if not X:
                continue

            X = np.array(X).astype(np.int32)
            z_data = np.random.randn(self.sample_size, self.z_dim)

            z_samples = self.Q(sess, X)

            _, loss, kl, recon = sess.run([\
                    self.vae_train_op, self.vae_loss, \
                    self.kl_loss, self.recon_loss\
                    ], feed_dict={
                        self.Q.X: X,
                        self.P.z: z_samples})

            """
            _, loss, kl, recon, z_samples, step = sess.run(
                    [self.vae_train_op, \
                            self.vae_loss, self.kl_loss, self.recon_loss, \
                            self.z_samples, self.global_step],\
                feed_dict={
                    self.X: X,
                    self.z: z_data,
                    })
            """
            if str(loss) == str(np.nan):
                print('[step/{}] early stop'.format(step))
                sys.exit()

            kl, recon = np.mean(kl), np.mean(recon)
            if step % self.summ_intv == 0:
                print('[step/{}] {} loss:{:.4} kl:{:.4} recon:{:.4}'.format(\
                        step, datetime.now(), loss, kl, recon))
                samples = sess.run(self.samples, feed_dict={self.z: z_data})
                docs = list(self.vocab_processor.reverse(samples.astype(np.int)))
                meta = {
                    'step': int(step),
                    'lr': float('{:.4}'.format(self.lr)),
                    'loss': float('{:.4}'.format(loss)),
                    'kl_loss': float('{:.4}'.format(kl)),
                    'recon_loss': float('{:.4}'.format(recon)),
                    'sample': docs
                    }
                self.to_json(meta, self.sample_path)
                self.saver.save(sess, self.model_dir, global_step=step)

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
        pass

if __name__ == '__main__':
    stex = StackEx()
    stex.train()
