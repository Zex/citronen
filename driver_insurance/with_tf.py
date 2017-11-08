# Porto Saguro driver insurance model
# Author: Zex Li <top_zlynch@yahoo.com>
import os
from datetime import datetime
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd


class DriveInsurance(object):

    Modes = ('train', 'validate', 'predict')

    def __init__(self, args):
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_dir = args.model_dir
        self.log_dir = os.path.join(
                self.model_dir,
                "log_{}th".format(datetime.today().timetuple().tm_yday))

        self.epochs = args.epochs
        self.lr = args.lr
        self.init_step = args.init_step
        self.summ_intv = args.summ_intv

        self.dropout_keep_prob = args.dropout

        self.prepare_dir()
        self.find_bondary()
        self._build_model()

    def mkdir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def prepare_dir(self):
       self.mkdir(self.model_dir) 
       self.mkdir(self.log_dir)

    def find_bondary(self):
        chunk = pd.read_csv(self.data_path, header=0, nrows=1)
        self.n_feature = chunk.shape[1]-2
        self.n_class = 2

    def gen_data(self):
        reader = pd.read_csv(self.data_path, header=0,
                chunksize=self.batch_size)
        for chunk in reader:
            yield self.batch_data(chunk)

    def batch_data(self, chunk):
        """id, X, y"""
        chunk = chunk.dropna()
        y = np.zero
        return chunkiloc[:,0], chunk.iloc[:,2:], chunk.iloc[:,1]

    def _build_model(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.n_feature], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.n_class], name="input_y")
        
        w = tf.get_variable('w', shape=[self.n_feature, self.n_class],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.n_class]), name='b')
        self.logits = tf.nn.xw_plus_b(
                tf.nn.dropout(self.input_x, self.dropout_keep_prob),
                w, b, name='logits')

        self.pred = tf.argmax(self.logits, 1, name="pred")
        self.loss = tf.nn.l2_loss(self.logits-self.input_y, name="loss")
        self.acc = tf.reduce_mean(tf.cast(
            tf.equal(self.pred, tf.argmax(self.input_y, 1)),
            "float"),
            name="acc")
        self.global_step = tf.Variable(self.init_step, trainable=False)
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(
                self.loss,
                global_step=self.global_step,
                name="train_op")

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("acc", self.acc)

        self.summary = tf.summary.merge_all()

    def foreach_train(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        self.saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        for i in range(self.epochs):
            self.foreach_epoch(sess)

    def foreach_epoch(self, sess):
        if self.mode ==  DriveInsurance.Mode[0]:
            for iid, x, y in self.gen_data():
                feed_dict = {
                    self.input_x: x,
                    self.input_y: y
                    }
                _, loss, acc, pred, step = tf.run(
                    [self.train_op, self.loss, self.acc, self.pred, self.global_step],
                    feed_dict=feed_dict)
                if step % self.summ_intv == 0:
                    print("[{}/{}] step:{} loss:{:.4f} acc:{:.4f} pred:{} lbl:{}".format(
                        self.mode, datetime.now(), step, loss, acc, pred, np.argmax(y, 1)), flush=True)
                    self.summary_writer.add_summary(summ, step)
                    self.saver.save(sess, self.model_dir,
                        global_step=tf.train.global_step(sess, self.global_step))
            
    def run(self):
        with tf.Session() as sess:
            self.foreach_train(sess)

    def __call__(self):
        self.run()


def init():
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--mode', default=DriveInsurance.Modes[0], type=str, help='Mode to run in', choices=DriveInsurance.Modes)
    parser.add_argument('--restore', default=False, action="store_true", help="Restore previous trained model")
    parser.add_argument('--data_path', default="./train.csv", type=str, help='Path to input data')
    parser.add_argument('--epochs', default=100000, type=int, help="Total epochs to train")
    parser.add_argument('--dropout', default=0.5, type=float, help="Dropout rate")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--summ_intv', default=500, type=int, help="Summary each several steps")
    parser.add_argument('--model_dir', default="../models/driver_insurance", type=str, help="Path to model and check point")
    parser.add_argument('--init_step', default=0, type=int, help="Initial training step")
    parser.add_argument('--max_doc', default=5000, type=int, help="Maximum document length")

    return parser.parse_args()

def start():
    args = init()
    DriveInsurance(args)()

if __name__ == '__main__':
    start()
