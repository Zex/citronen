#!/usr/bin/env python3
# Text classification with TF
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import re
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from subject_encoder import load_l2table
import tensorflow as tf
from tensorflow.contrib import learn, layers, framework
from sklearn.utils import shuffle

import nltk.data
from nltk import wordpunct_tokenize
from nltk.tag import pos_tag

nltk.data.path.insert(0, "/media/sf_patsnap/nltk_data")
expected_types = ("NNP", "NN", "NNS", "JJ")


class SD(object):

    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.l2table = load_l2table()
        self.class_map = list(set(self.l2table.values()))
        self.max_doc_len = 300
        #self.find_bondary()
        self.train_vocab()

    def tokenize_text(self, text):
        tokens = []
        for doc in text:
            tokens.append([w for w, t in pos_tag(wordpunct_tokenize(doc)) if t in expected_types])
        return tokens

    def find_bondary(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0, 
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            text, _ = self.extract_xy(chunk)
            tokens = self.tokenize_text(text)
            self.max_doc_len = \
                    np.max([self.max_doc_len, np.max([len(t) for t in tokens])])
        print("max doc len: {}".format(self.max_doc_len))

    def train_vocab(self):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_doc_len)
        reader = pd.read_csv(self.data_path, engine='python', header=0, 
            delimiter="###", chunksize=self.batch_size)
        [self.process_chunk(*self.extract_xy(chunk)) for chunk in reader]

    def gen_data(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0, 
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            yield self.process_chunk(*self.extract_xy(chunk))

    def extract_xy(self, chunk):
        chunk = shuffle(chunk.dropna().replace({"subcate":self.l2table}))

        text = chunk["desc"]
        label = chunk["subcate"]

        return text, label

    def process_chunk(self, text, label):
        """
        np.random.seed(17)
        indices = np.random.permutation(np.arange(len(label)))
        text = text[indices]#.tolist()
        label = label[indices]#.tolist()
        """

        #x = list(self.vocab_processor.fit_transform(text))
        tokens = self.tokenize_text(text)
        x = list(self.vocab_processor.fit_transform(
            [' '.join(t) for t in tokens]
            ))

        y = []
        for l in label:
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(l)] = 1.
            y.append(one)
        return x, np.array(y)


class Springer(object):

    def __init__(self, args):
        super(Springer, self).__init__()
        self.sd = SD(args)

        # Train args
        self.model_dir = args.model_dir
        self.data_path = args.data_path
        self.epochs = args.epochs
        self.dropout_rate = args.dropout
        self.init_step = args.init_step
        self.lr = args.lr

        # Model args
        self.total_class = len(self.sd.class_map)
        self.seqlen = self.sd.max_doc_len
        self.embed_dim = 128 
        self.total_filters = 128
        self.filter_sizes = [3, 5]

        self.prepare_dir()
        self._build_model()

    def prepare_dir(self):
        self.log_path = os.path.join(self.model_dir, "log")
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        self.vocab_path = os.path.join(self.model_dir, "vocab")
        self.sd.vocab_processor.save(self.vocab_path)
        self.vocab_size = len(self.sd.vocab_processor.vocabulary_)
        print("total vocab: {}".format(self.vocab_size))

    def _build_model(self):

        # Define model
        self.input_x = tf.placeholder(tf.int32, [None, self.seqlen], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.total_class], name="input_y")

        loss = tf.constant(0.0)

        self.w = tf.Variable(tf.random_uniform([self.vocab_size+1, self.embed_dim], -1.0, 1.0), name="w_em")
        self.embed_chars = tf.nn.embedding_lookup(self.w, self.input_x)
        self.embed = tf.expand_dims(self.embed_chars, -1)
        #self.embed = tf.contrib.layers.embed_sequence(
        #    self.input_x,
        #    vocab_size=self.vocab_size,
        #    embed_dim=self.embed_dim)

        self.pools = []

        for i, filter_sz in enumerate(self.filter_sizes):
            filter_shape = [filter_sz, self.embed_dim, 1, self.total_filters]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
            b = tf.Variable(tf.constant(0.1, shape=[self.total_filters]), name="b")
            conv = tf.nn.conv2d(self.embed,
                                w,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv")
            hidden = tf.nn.relu(tf.nn.bias_add(conv, b), name="hidden")
            pool = tf.nn.max_pool(hidden,
                                ksize=[1, self.seqlen-filter_sz+1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="pool")
            self.pools.append(pool)
        
        filter_comb = self.total_filters * len(self.filter_sizes)
        self.hidden_pool = tf.concat(self.pools, 3)
        self.hidden_flat = tf.reshape(self.hidden_pool, [-1, filter_comb])

        self.hidden_dropout = tf.nn.dropout(self.hidden_flat, self.dropout_rate)

        #w = tf.get_variable("w", shape=[filter_comb, self.total_class],
        #                    initializer=tf.contrib.layers.xavier_initializer())
        #b = tf.Variable(tf.constant(0.1, shape=[self.total_class]), name="b")
        #loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        #self.logits = tf.nn.xw_plus_b(self.hidden_dropout, w, b, name="logits")
        self.logits = layers.fully_connected(self.hidden_dropout, self.total_class)
        self.pred = tf.argmax(self.logits, 1, name="pred")

        #self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y, name="loss")
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits)
        #self.loss = tf.reduce_mean(losses) + 0.384*loss

        corr = tf.equal(self.pred, tf.argmax(self.input_y, 1))
        self.acc = tf.reduce_mean(tf.cast(corr, "float"), name="acc")

        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        """
        opt = tf.train.AdamOptimizer(self.lr)
        params = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(
                params,
                global_step=self.global_step)
        """
        self.train_op = layers.optimize_loss(
                self.loss,
                self.global_step,
                #framework.get_global_step(),
                optimizer="Adam",
                learning_rate=self.lr)
        """
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss,
                global_step=self.global_step)
        """

        # Define summary
        summary = []
        for k, v in params:
            summary.append(tf.summary.histogram(v.name, k))
            summary.append(tf.summary.scalar(v.name, tf.nn.zero_fraction(k)))

        summary.append(tf.summary.scalar("loss", self.loss))
        summary.append(tf.summary.scalar("acc", self.acc))

        self.summary = tf.summary.merge(summary)
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self):
        with tf.Session() as sess:
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                self.foreach_epoch(sess)
                self.saver.save(sess, self.model_dir,
                        global_step=tf.train.global_step(sess, self.global_step))

    def foreach_epoch(self, sess):
        for i, (x, y) in enumerate(self.sd.gen_data()):
            _, step, summ, loss, acc, pred = sess.run(
                    [self.train_op, self.global_step, self.summary, \
                            self.loss, self.acc, self.pred],
                    feed_dict={
                        self.input_x: x,
                        self.input_y: y,
                    })
            self.summary_writer.add_summary(summ, step)
            print("{}: step:{} loss:{:.4f} acc:{:.4f} pred:{} lbl:{}".format(
                datetime.now(), step, loss, acc, pred, np.argmax(y, 1)), flush=True)

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    parser.add_argument('--chunk_size', default=32, type=int, help='Load data by size of chunk')
    parser.add_argument('--data_path', default="../data/springer/mini.csv", type=str, help='Path to input data')
    parser.add_argument('--epochs', default=1000, type=int, help="Total epochs to train")
    parser.add_argument('--dropout', default=0.5, type=int, help="Dropout rate")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=float, help="Batch size")
    parser.add_argument('--model_dir', default="../models/springer", type=str, help="Path to model and check point")
    parser.add_argument('--init_step', default=0, type=int, help="Initial training step")

    args = parser.parse_args()
    return args

def start():
    args = init()
    springer = Springer(args)
    springer.train()

if __name__ == '__main__':
    start()
