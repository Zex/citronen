#!/usr/bin/env python3
# Text classification with TF
# Author: Zex Li <top_zlynch@yahoo.com>
import os, sys, re, glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn, layers, framework
from sklearn.utils import shuffle
from data_helper import load_d3table, extract_xy
from data_helper import level_decode


class DataProvider(object):
    """Help prepare data for training, validation and prediction"""
    def __init__(self, args):
        super(DataProvider, self).__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.d3table = load_d3table()
        self.class_map = self.d3table.drop_duplicates().values.squeeze().tolist()
        self.max_doc = args.max_doc
        self.vocab_path = os.path.join(args.model_dir, "vocab")
        if True:
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_doc)
            self.vocab_processor.save(self.vocab_path)
            self.x, self.y = self.load_data()
        print("Max document length: {}".format(self.max_doc))

    def batch_data(self):
        total_batch = int((len(self.x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield self.x[current:current+self.batch_size+1], \
                        self.y[current:current+self.batch_size+1]

    def load_data(self):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="#")
        chunk = shuffle(chunk)
        return self.process_chunk(*extract_xy(chunk))

    def gen_data(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="#", chunksize=self.batch_size)
        for chunk in reader:
            chunk = shuffle(chunk)
            yield self.process_chunk(*extract_xy(chunk))

    def process_chunk(self, text, label):
        x = np.array(list(self.vocab_processor.fit_transform(text)))
        y = []
        for l in label:
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(l)] = 1.
            y.append(one)
        return x, np.array(y)

class Naics(object):
    """Define the model
    """

    Modes = ('train', 'validate', 'predict')

    def __init__(self, args):
        super(Naics, self).__init__()
        self.sd = DataProvider(args)
        # Train args
        self.model_dir = args.model_dir
        self.data_path = args.data_path
        self.epochs = args.epochs
        self.dropout_rate = args.dropout
        self.clip_norm = args.clip_norm
        self.init_step = args.init_step
        self.restore = args.restore
        self.mode = args.mode
        self.lr = args.lr
        # Model args
        self.total_class = len(self.provider.class_map)
        self.seqlen = self.provider.max_doc
        self.embed_dim = 128
        self.total_filters = 128
        self.total_layer = 5
        self.filter_sizes = [3, 5]

        self.prepare_dir()

    def prepare_dir(self):
        self.log_path = os.path.join(
                self.model_dir,
                "log_{}th".format(datetime.today().timetuple().tm_yday))
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)
        #self.vocab_size = len(self.provider.global_tokens)
        self.vocab_size = len(self.provider.vocab_processor.vocabulary_)
        print("total vocab: {}".format(self.vocab_size))

        self.summary = tf.summary.merge(summary)

    def _build_model(self):
        # Define model
        self.input_x = tf.placeholder(tf.int32, [None, self.seqlen], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.total_class], name="input_y")

        self.w_em = tf.Variable(tf.random_uniform(
                    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                name="w_em_{}".format(0))
        self.embed_chars = tf.nn.embedding_lookup(self.w_em, self.input_x)
        self.embed = tf.expand_dims(self.embed_chars, -1)

        self.pools = []

        for i, filter_sz in enumerate(self.filter_sizes):
            filter_shape = [filter_sz, self.embed_dim, 1, self.total_filters]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w_{}".format(i))
            b = tf.Variable(tf.constant(0.1, shape=[self.total_filters]), name="b_{}".format(i))
            conv = tf.nn.conv2d(self.embed,
                                w,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv_{}".format(i))
            hidden = tf.nn.relu(tf.nn.bias_add(conv, b), name="hidden_{}".format(i))
            pool = tf.nn.max_pool(hidden,
                                ksize=[1, self.seqlen-filter_sz+1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="pool_{}".format(i))
            self.pools.append(pool)

        filter_comb = self.total_filters * len(self.filter_sizes)
        self.hidden_pool = tf.concat(self.pools, 3)
        self.hidden_flat = tf.reshape(self.hidden_pool, [-1, filter_comb])

        self.hidden_dropout = tf.nn.dropout(self.hidden_flat, self.dropout_rate)

        w = tf.get_variable("w", shape=[filter_comb, self.total_class],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.total_class]), name="b")
        #loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
        self.logits = tf.nn.xw_plus_b(self.hidden_dropout, w, b, name="logits")
        self.pred = tf.argmax(self.logits, 1, name="pred")

        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.input_y),
                name="loss"
                )

        corr = tf.equal(self.pred, tf.argmax(self.input_y, 1), name="corr")
        self.acc = tf.reduce_mean(tf.cast(corr, "float"), name="acc")

        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(self.lr)
        params = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(
                params,
                global_step=self.global_step,
                name="train_op")

        # Define summary
        summary = []
        for k, v in params:
            summary.append(tf.summary.histogram(v.name, k))
            summary.append(tf.summary.scalar(v.name, tf.nn.zero_fraction(k)))

        summary.append(tf.summary.scalar("loss", self.loss))
        summary.append(tf.summary.scalar("acc", self.acc))

        self.summary = tf.summary.merge(summary, name="merge_summary")

    def _restore_model(self, sess):
        if not self.graph_path:
            print("Pretrained model not found")
            sys.exit(1)
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.import_meta_graph(self.graph_path)
        self.saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(self.model_dir)))
        graph = tf.get_default_graph()
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
                tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        print("Found {} variables".format(len(global_vars)))

        self.input_x = graph.get_tensor_by_name("input_x:0")
        self.input_y = graph.get_tensor_by_name("input_y:0")
        self.pred = graph.get_tensor_by_name("pred:0")

        if self.mode != Naics.Modes[2]:
            self.global_step = graph.get_tensor_by_name("global_step:0")
            self.loss = tf.reduce_mean(graph.get_tensor_by_name("loss_1:0"))
            self.acc = graph.get_tensor_by_name("acc_1:0")
            self.train_op = graph.get_tensor_by_name("train_op:0").op
            self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))


    def run(self):
        with tf.Session() as sess:
            if self.restore:
                metas = sorted(glob.glob("{}-*meta".format(self.model_dir)), key=os.path.getmtime)
                self.graph_path = metas[-1] if metas else None
                self._restore_model(sess)
            else:
                self._build_model()

            if self.mode == Naics.Modes[2]:
                self.foreach_epoch(sess)
            elif self.mode == Naics.Modes[1]:
                self.epochs = 1
                self.foreach_train(sess)
            else:
                self.foreach_train(sess)

    def foreach_train(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
        self.saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        for epoch in range(self.epochs):
            self.foreach_epoch(sess)
            self.saver.save(sess, self.model_dir,
                        global_step=tf.train.global_step(sess, self.global_step),
                        max_to_keep=2)

    def foreach_epoch(self, sess):
        # for x, y in self.provider.gen_data():
        for x, y in self.provider.batch_data():
            feed_dict = {
                    self.input_x: x,
                    self.input_y: y,
            }
            if self.mode == Naics.Modes[2]: # predict
                pred = sess.run([self.pred], feed_dict=feed_dict)
                [print("[{}/{}] naics code: {} naics name: {}".format(
                    self.mode, datetime.now(),
                    *level_decode(
                        p,
                        d3table=self.provider.d3table,
                        class_map=self.provider.class_map
                        )), flush=True) for p in np.squeeze(pred)]
            elif self.mode == Naics.Modes[1]: # evaluate
                summ, loss, acc, pred = sess.run(
                    [self.summary, self.loss, self.acc, self.pred],
                    feed_dict=feed_dict)
                self.summary_writer.add_summary(summ, step)
                print("[{}/{}] loss:{:.4f} acc:{:.4f} pred:{} lbl:{}".format(
                    self.mode, datetime.now(), loss, acc, pred, np.argmax(y, 1)), flush=True)
            else: # train
                _, step, summ, loss, acc, pred = sess.run(
                    [self.train_op, self.global_step, self.summary, \
                            self.loss, self.acc, self.pred],
                    feed_dict=feed_dict)
                self.summary_writer.add_summary(summ, step)
                if step % 1000 == 0:
                    print("[{}/{}] step:{} loss:{:.4f} acc:{:.4f} pred:{} lbl:{}".format(
                        self.mode, datetime.now(), step, loss, acc, pred, np.argmax(y, 1)), flush=True)


def init():
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=Naics.Modes[0], type=str, help='Mode to run in', choices=Naics.Modes)
    parser.add_argument('--restore', default=False, action="store_true", help="Restore previous trained model")
    parser.add_argument('--data_path', default="../data/naics/mini.csv", type=str, help='Path to input data')
    parser.add_argument('--epochs', default=10000, type=int, help="Total epochs to train")
    parser.add_argument('--dropout', default=0.5, type=float, help="Dropout rate")
    parser.add_argument('--clip_norm', default=5.0, type=int, help="Gradient clipping ratio")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--model_dir', default="../models/naics", type=str, help="Path to model and check point")
    parser.add_argument('--init_step', default=0, type=int, help="Initial training step")
    parser.add_argument('--max_doc', default=5000, type=int, help="Maximum document length")

    args = parser.parse_args()
    return args

def start():
    args = init()
    naics = Naics(args)
    naics.run()

if __name__ == '__main__':
    start()
