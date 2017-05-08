
# RUSSIAN HOUSING DATA MODEL
# Author: Zex <top_zlynch@yahoo.com>

from common import mean_squared_error
from pandas import read_csv
import tensorflow as tf
from tensorflow import gfile
import argparse
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)

class RusModel:
  modes = ['train', 'eval', 'test']

  def __init__(self, args):
    self.model_id = args.model_id
    self.update_path()
    self.batch_size = args.batch_size
    self.max_features = 290
    self.lr = args.lr
    self.mode = args.mode
    self.init_epoch = args.init_epoch
    self.total_epochs = args.epochs
    self.steps_per_epoch = args.steps
    self.master = args.master

    self.create_model()

  def update_path(self):
    self.model_path = "../models/{}.h5".format(self.model_id)
    self.chkpt_path = "../models/{}.chkpt".format(self.model_id)
    self.logdir = "../build/{}.log".format(self.model_id)
    self.prep_data_path = '../build/{}-prep.csv'.format(self.model_id)
    self.data_base = '../data/russian_housing'
    self.train_data_path = "{}/train.csv".format(self.data_base)
    self.eval_data_path = "{}/eval.csv".format(self.data_base)
    self.test_data_path = "{}/test.csv".format(self.data_base)

  def create_model(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    self.feat_input = tf.placeholder(tf.float64, shape=(self.batch_size, self.max_features), name='feat_input')
    self.labels = tf.placeholder(tf.float64, shape=(self.batch_size,), name='labels')
    #self.feat_input = tf.truncated_normal(self.feat_input.shape, dtype=tf.float64)
    #self.labels = tf.truncated_normal(self.labels.shape, dtype=tf.float64)

    self.weights = tf.Variable(tf.truncated_normal([self.max_features, self.batch_size], dtype=tf.float64), name='weights', trainable=True)
    self.bias = tf.Variable(tf.truncated_normal([self.batch_size, self.batch_size], dtype=tf.float64), name='bias', trainable=True)

    self.logits = tf.add(tf.matmul(self.feat_input, self.weights), self.bias)
    self.cost = tf.reduce_sum(tf.pow(self.logits - self.labels, 2))/(2 * self.batch_size)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        self.lr,        # initial learning rate
        global_step,    # current index
        self.batch_size,# decay step
        0.95,           # decay rate
        staircase=True)

    self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
    tf.summary.scalar('cost', self.cost)
    self.saver = tf.train.Saver(tf.global_variables())

    return self.optimizer

  def save(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      print('model saved @ {}'.format(self.saver.save(sess, self.model_path)))

  def load(self):
    if not gfile.Exists(self.model_path):
      return False
    with tf.Graph().as_default(), tf.Session() as sess:
      self.saver.restore(sess, self.model_path)
    return True

  def train(self):
    if not gfile.Exists(self.prep_data_path):
      preprocess(self.train_data_path)
    g = tf.get_default_graph()
    print('ver:{} '.format(g.version ))
    if True:
      with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run(session=sess)
        for e in range(self.total_epochs):
          self.one_epoch(sess)
          print('chkpt saved @ {}'.format(self.saver.save(sess, self.chkpt_path, global_step=e)))

  def one_epoch(self, sess):
    # TODO
    #sv = tf.train.Supervisor(logdir=self.logdir)
    #with sv.managed_session(self.master) as sess:
    if True:
      for step, (x, y) in enumerate(generate_data(self.prep_data_path, self.batch_size)):
        #if sv.should_stop():
        #  break
        if x.shape[0] < self.batch_size:
          x = np.resize(x, (self.batch_size, self.max_features))
          y = np.resize(y, (self.batch_size,))
        x, y = x.astype(np.float64).tolist(), y.astype(np.float64).tolist()
        sess.run([self.optimizer], feed_dict={
         self.feat_input: x,
         self.labels: y,
        })
        if step > self.steps_per_epoch:
          break
        logits = sess.run([self.logits], feed_dict={
         self.feat_input: x,
         self.labels: y,
        })
        if logits and len(logits) > 0:
          print(logits)
          loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
          tf.summary.scalar('sparse_softmax_cross_entropy_with_logits', loss)

  def eval(self, eval_data, eval_labels):
    sv = tf.train.Supervisor(logdir=self.logdir)
    with sv.managed_session(self.master) as sess:
      tf.global_variables_initializer().run(session=sess)
      eval_pred = None #TODO
      pred = sess.run(eval_pred, feed_dict={
        feat_input: eval_data,
        labels: eval_labels
        })
      print('pred:', pred)
    return pred

  def test(self):
    pass

def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=RusModel.modes)
  parser.add_argument('--model_id', default='rustf', type=str, help='Prefix for model persistance')
  parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
  parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
  parser.add_argument('--steps', default=1000, type=int, help='Number of steps in one epoch')
  parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
  parser.add_argument('--master', default='', type=str, help='Master for distributed execution')
  args = parser.parse_args()
  return args

def preprocess(source, chunksize):
  reader = read_csv(source, header=0, chunksize=chunksize)
  if gfile.Exists(prep_data_path):
    gfile.Remove(prep_data_path)

  for data in reader:
    data = data.fillna(0)
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    product_type, sub_area, ecology = \
      data['product_type'].values, data['sub_area'].values, data['ecology'].values
    data['product_type'] = np.reshape([one_hot(
          x, n=np.unique(product_type).shape[0]+1, filters='') for x in product_type], product_type.shape)
    sub_area = np.array([s.replace(' ', '').replace('-', '').replace('\'', '').replace(',', '') for s in sub_area])
    data['sub_area'] = np.reshape([one_hot(
      x, n=np.unique(sub_area).shape[0]+1) for x in sub_area], sub_area.shape)
    ecology = np.array([s.replace(' ', '').replace('-', '').replace('\'', '').replace(',', '') for s in ecology])
    data['ecology'] = np.reshape([one_hot(
      x, n=np.unique(ecology).shape[0]+1) for x in ecology], ecology.shape)
    data.to_csv(prep_data_path) if not isfile(prep_data_path) else data.to_csv(prep_data_path, mode='a', header=False)

def load_chunk(chunk):
  chunk = chunk.fillna(0)
  ids, y = chunk['id'], chunk['price_doc'].values
  chunk['timestamp'] = chunk['timestamp'].values.astype('datetime64').astype(np.int)
  x = chunk.iloc[0:chunk.shape[0], 1:291].values
  x = x.astype(np.float64)
  y = y.astype(np.float64)
  #y = normalize(y)
  return x, y

def generate_data(source, chunksize=64, forever=True): 
  while True:
    reader = read_csv(source, header=0, chunksize=chunksize)
    for chunk in reader:
      x, y = load_chunk(chunk)
      yield x, y
    if not forever: break

def main(_):
  args = init()
  graph = RusModel(args)
#  if not graph.load():
#  graph.create_model()
  graph.train()

if __name__ == '__main__':
  tf.app.run(main=main)

