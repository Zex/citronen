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
from data_helper import load_l2table, load_l1table, tokenize_text, extract_xy
from data_helper import train_vocab, level_decode, load_vocab


class SD(object):
    """Help prepare data for training, validation and prediction"""
    def __init__(self, args, need_shuffle=True):
        super(SD, self).__init__()
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.l1table = load_l1table()
        self.l2table = load_l2table()
#        self.global_tokens = load_global_tokens()
        self.class_map = list(set(self.l2table.values()))
        self.total_class = len(self.class_map)
        self.max_doc = args.max_doc
        self.vocab_path = args.vocab_path if args.vocab_path else os.path.join(args.model_dir,
                "vocab_{}th".format(datetime.today().timetuple().tm_yday))
#       self.hv = get_hashing_vec(self.max_doc, "english")
#       self.find_bondary()
        if os.path.isfile(self.vocab_path):
            self.vocab_processor = load_vocab(self.vocab_path)
        else:
            self.vocab_processor = train_vocab(self.data_path, self.vocab_path, self.max_doc)
        self.x, self.y = self.load_data(need_shuffle)
        print("Max document length: {}".format(self.max_doc))

    def find_bondary(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            text, _, _ = extract_xy(chunk)
            tokens = tokenize_text(text)
            self.max_doc = \
                    np.max([self.max_doc, np.max([len(t) for t in tokens])])

    def batch_data(self):
        total_batch = int((len(self.x)-1)/self.batch_size)+1
        print("Total batch: {}".format(total_batch))
        for i in range(total_batch):
            current = i * self.batch_size
            yield self.x[current:current+self.batch_size+1], \
                        self.y[current:current+self.batch_size+1]

    def load_data(self, need_shuffle=True):
        chunk = pd.read_csv(self.data_path, engine='python', header=0, delimiter="###")
        if need_shuffle:
            chunk = shuffle(chunk)
        return self.process_chunk(*extract_xy(chunk, l2table=self.l2table))

    def gen_data(self, need_shuffle=True):
        reader = pd.read_csv(self.data_path, engine='python', header=0,
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            if need_shuffle:
                chunk = shuffle(chunk)
            yield self.process_chunk(*extract_xy(chunk, l2table=self.l2table))

    def process_chunk(self, text, label1, label):
        #x = self.hv.transform(text).toarray()#[' '.join(t) for t in tokens]).toarray()
        y = []
        def init_label(lbl):
            one = np.zeros(len(self.class_map))
            one[self.class_map.index(lbl)] = 1.
            y.append(one)
            #tf.one_hot(self.class_map.index(lbl), self.total_class))

        x = list(self.vocab_processor.transform(text))
        list(map(init_label, label))
        #[init_label(lbl) for lbl in label]
        #np.vectorize(init_label)(label)
        return x, y

class Springer(object):
    """Define the model
    """

    Modes = ('train', 'validate', 'predict')

    def __init__(self, args):
        super(Springer, self).__init__()
        # Train args
        self.epochs = args.epochs
        self.summ_intv = args.summ_intv
        self.verbose = args.verbose
        self.dropout = args.dropout
        self.clip_norm = args.clip_norm
        self.init_step = args.init_step
        self.restore = args.restore
        self.mode = args.mode
        self.lr = args.lr
        self.model_dir = args.model_dir
        self.data_path = args.data_path
        self.pred_output = args.pred_output_path

        if not self.mode == Springer.Modes[2]: # predict
            self.sd = SD(args)
        else:
            self.sd = SD(args, False)
        # Model args
        self.total_class = self.sd.total_class
        self.seqlen = self.sd.max_doc
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
        #self.vocab_size = len(self.sd.global_tokens)
        self.vocab_size = len(self.sd.vocab_processor.vocabulary_)
        print("total vocab: {}".format(self.vocab_size))

    def __build_model(self):
        # Define model
        self.input_x = tf.placeholder(tf.int32, [None, self.seqlen], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.total_class], name="input_y")

        w = tf.get_variable("w_e", [self.seqlen, self.embed_dim])
        #self.embed = tf.nn.embedding_lookup(w, self.input_x)
        self.embed = layers.embed_sequence(self.input_x, vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        #self.rnn_unit = tf.nn.rnn_cell.DropoutWrapper(
        self.rnn_unit = tf.nn.rnn_cell.GRUCell(self.embed_dim)
        #        output_keep_prob=1-self.dropout_keep)
        #self.cell_stack = tf.nn.rnn_cell.MultiRNNCell([self.rnn_unit] * self.total_layer)
        words = tf.unstack(self.embed, axis=1)
        _, encoding = tf.nn.static_rnn(
                cell=self.rnn_unit, inputs=words,
                dtype=tf.float32)
        # calc logits
        self.logits = tf.layers.dense(encoding, self.total_class, activation=None)

        self.pred = tf.nn.softmax(self.logits)
        #self.pred = tf.argmax(self.logits, 1, name="pred")
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, 1),
                            tf.argmax(self.logits, 1)), "float"), name="acc")
        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))

        params, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, tf.trainable_variables()),
                self.clip_norm)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(
                zip(params, tf.trainable_variables()))

        self.global_step = tf.Variable(self.init_step, trainable=False)

        summary = []
        summary.append(tf.summary.scalar("loss", self.loss))
        summary.append(tf.summary.scalar("acc", self.acc))

        self.summary = tf.summary.merge(summary)


    def get_logits(self, name):
        w_em = tf.Variable(tf.random_uniform(
                    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                    name="w_em_{}_{}".format(name, 0))
        embed = tf.expand_dims(
            tf.nn.embedding_lookup(w_em, self.input_x),
            -1)

        pools = []
        for i, filter_sz in enumerate(self.filter_sizes):
            filter_shape = [filter_sz, self.embed_dim, 1, self.total_filters]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w_{}_{}".format(name, i))
            b = tf.Variable(tf.constant(0.1, shape=[self.total_filters]), name="b_{}_{}".format(name, i))
            conv = tf.nn.conv2d(embed,
                                w,
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="conv_{}_{}".format(name, i))
            hidden = tf.nn.relu(tf.nn.bias_add(conv, b), name="hidden_{}_{}".format(name, i))
            pool = tf.nn.max_pool(hidden,
                                ksize=[1, self.seqlen-filter_sz+1, 1, 1],
                                strides=[1, 1, 1, 1],
                                padding="VALID",
                                name="pool_{}_{}".format(name, i))
            pools.append(pool)

        filter_comb = self.total_filters * len(self.filter_sizes)
        hidden_flat = tf.reshape(
                tf.concat(pools, 3),
                [-1, filter_comb])

        hidden_dropout = tf.nn.dropout(hidden_flat, self.dropout_keep)

        w = tf.get_variable("w", shape=[filter_comb, self.total_class],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.total_class]), name="b")
        return tf.nn.xw_plus_b(hidden_dropout, w, b, name="logits_{}".format(name))

    def _build_model(self):
        # Define model
        self.input_x = tf.placeholder(tf.int32, [None, self.seqlen], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.total_class], name="input_y")
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")

        self.logits = self.get_logits('springer')
        self.pred = tf.argmax(self.logits, 1, name="pred")

        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.input_y),
                name="loss"
                )

        self.acc = tf.reduce_mean(tf.cast(
            tf.equal(self.pred, tf.argmax(self.input_y, 1), name="corr"),
            "float"), name="acc")

        self.global_step = tf.Variable(self.init_step, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(self.lr)
        params = opt.compute_gradients(self.loss)
        self.train_op = opt.apply_gradients(
                params,
                global_step=self.global_step,
                name="train_op")
        """
        self.train_op = layers.optimize_loss(
                self.loss,
                self.global_step,
                #framework.get_global_step(),
                optimizer="Adam",
                learning_rate=self.lr)
        """
        """
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                self.loss,
                global_step=self.global_step)
        """

        # Define summary
        summary = []
        for k, v in params:
            summary.append(tf.summary.histogram(v.name.replace(':', '_'), k))
            summary.append(tf.summary.scalar(v.name.replace(':', '_'), tf.nn.zero_fraction(k)))

        summary.append(tf.summary.scalar("loss", self.loss))
        summary.append(tf.summary.scalar("acc", self.acc))

        self.summary = tf.summary.merge(summary, name="merge_summary")

    def _restore_model(self, sess):
        if not self.graph_path:
            print("Pretrained model not found")
            sys.exit(1)
        self.saver = tf.train.import_meta_graph(self.graph_path)
        self.saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(self.model_dir)))
        graph = tf.get_default_graph()
        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
                tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        print("Found {} variables".format(len(global_vars)))

        self.input_x = graph.get_tensor_by_name("input_x:0")
        self.dropout_keep = graph.get_tensor_by_name("dropout_keep:0")
        self.pred = graph.get_tensor_by_name("pred:0")

    def run(self):
        with tf.Session() as sess:
            if self.restore:
                metas = sorted(glob.glob("{}/*meta".format(self.model_dir)), key=os.path.getmtime)
                self.graph_path = metas[-1] if metas else None
                self._restore_model(sess)
            else:
                self._build_model()

            if self.mode == Springer.Modes[2]:
                if os.path.isfile(self.pred_output):
                    os.remove(self.pred_output)
                self.foreach_epoch(sess)
            elif self.mode == Springer.Modes[1]:
                self.foreach_epoch(sess)
            else:
                self.foreach_train(sess)

    def foreach_train(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
        self.saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        for epoch in range(self.epochs):
            self.foreach_epoch(sess)

    def foreach_epoch(self, sess):
        # for x, y in self.sd.gen_data():
        for x, y in self.sd.batch_data():
            feed_dict = {
                    self.input_x: x,
                    self.dropout_keep: self.dropout,
            }
            if self.mode == Springer.Modes[2]: # predict
                pred = sess.run([self.pred], feed_dict=feed_dict)
                header = ['iid', 'l1', 'l2']
                df = pd.DataFrame(columns=header)
                pred = np.squeeze(pred)
                for p in pred:
                    iid, l1, l2 = level_decode(
                        p,
                        l1table=self.sd.l1table,
                        l2table=self.sd.l2table,
                        class_map=self.sd.class_map)
                    df = df.append(pd.Series((iid, l1, l2), index=header), ignore_index=True)
                if os.path.isfile(self.pred_output):
                    df.to_csv(self.pred_output, header=True, index=False, sep='#', mode='a')
                else:
                    df.to_csv(self.pred_output, header=False, index=False, sep='#')
            elif self.mode == Springer.Modes[1]: # evaluate
                pred = sess.run([self.pred], feed_dict=feed_dict)
#                self.summary_writer.add_summary(summ, step)
                y, pred = np.argmax(y, 1), np.squeeze(pred)
                acc = np.float(np.sum(pred == y))/np.float(len(y))
                if self.verbose:
                    print("[{}/{}] total:{} acc:{:.4g} pred:{} lbl:{}".format(
                        self.mode, datetime.now(), len(y), acc, pred, y), flush=True)
                else:
                    print("[{}/{}] total:{} acc:{:.4g}".format(
                        self.mode, datetime.now(), len(y), acc), flush=True)
            else: # train
                feed_dict.update({
                    self.input_y: y,
                })
                _, step, summ, loss, acc, pred = sess.run(
                    [self.train_op, self.global_step, self.summary, \
                            self.loss, self.acc, self.pred],
                    feed_dict=feed_dict)
                if step % self.summ_intv == 0:
                    if self.verbose:
                        print("[{}/{}] step:{} loss:{:.4f} acc:{:.4f} pred:{} lbl:{}".format(
                            self.mode, datetime.now(), step, loss, acc, pred, np.argmax(y, 1)), flush=True)
                    else:
                        print("[{}/{}] step:{} loss:{:.4f} acc:{:.4f}".format(
                            self.mode, datetime.now(), step, loss, acc), flush=True)
                    self.summary_writer.add_summary(summ, step)
                    self.saver.save(sess, self.model_dir,
                        global_step=tf.train.global_step(sess, self.global_step))


def init():
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default=Springer.Modes[0], type=str, help='Mode to run in', choices=Springer.Modes)
    parser.add_argument('--restore', default=False, action="store_true", help="Restore previous trained model")
    parser.add_argument('--data_path', default="../data/springer/mini.csv", type=str, help='Path to input data')
    parser.add_argument('--vocab_path', default=None, type=str, help='Path to input data')
    parser.add_argument('--pred_output_path', default="../data/springer/predict.csv", type=str, help='Path to prediction data ouput')
    parser.add_argument('--epochs', default=100000, type=int, help="Total epochs to train")
    parser.add_argument('--dropout', default=0.5, type=float, help="Dropout keep prob")
    parser.add_argument('--clip_norm', default=5.0, type=int, help="Gradient clipping ratio")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--summ_intv', default=500, type=int, help="Summary each several steps")
    parser.add_argument('--verbose', default=False, action='store_true', help="Print verbose")
    parser.add_argument('--model_dir', default="../models/springer", type=str, help="Path to model and check point")
    parser.add_argument('--init_step', default=0, type=int, help="Initial training step")
    parser.add_argument('--max_doc', default=5000, type=int, help="Maximum document length")

    return parser.parse_args()

def start():
    args = init()
    springer = Springer(args)
    springer.run()

if __name__ == '__main__':
    start()
