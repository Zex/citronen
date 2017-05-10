
# RUSSIAN HOUSING DATA MODEL
# Author: Zex <top_zlynch@yahoo.com>

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from common import mean_squared_error
from pandas import read_csv
import tensorflow as tf
from tensorflow import gfile
import argparse
import numpy as np

fig = None

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
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self.learning_rate = tf.Variable(self.lr, dtype=tf.float64) 
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
    self.labels = tf.placeholder(tf.float64, shape=(self.batch_size, 1), name='labels')
    self.feat_input = tf.truncated_normal(self.feat_input.shape, dtype=tf.float64)
    self.labels = tf.truncated_normal(self.labels.shape, dtype=tf.float64)
    print('[feat_input] {} [labels] {}'.format(self.feat_input, self.labels))

    # hidden 0
    self.weights_0 = tf.Variable(tf.truncated_normal([self.max_features, 1], dtype=tf.float64), name='w_0', trainable=True)
    self.bias_0 = tf.Variable(tf.truncated_normal([self.batch_size, 1], dtype=tf.float64), name='b_0', trainable=True)
    self.hidden_0 = tf.add(tf.matmul(self.feat_input, self.weights_0), self.bias_0)
    self.hidden_0 = tf.nn.relu(self.hidden_0)
    print('[hidden] {}'.format(self.hidden_0))

#    # hidden 1
#    self.weights_1 = tf.Variable(tf.truncated_normal([self.max_features, 1], dtype=tf.float64), name='w_1', trainable=True)
#    self.bias_1 = tf.Variable(tf.truncated_normal([self.max_features, 1], dtype=tf.float64), name='b_1', trainable=True)
#    self.hidden_1 = tf.add(tf.matmul(self.hidden_0, self.weights_1), self.bias_1)
#    self.hidden_1 = tf.nn.relu(self.hidden_1)
#    print('[hidden] {}'.format(self.hidden_1))
#
#    # hidden 2
#    self.weights_2 = tf.Variable(tf.truncated_normal([1, self.batch_size], dtype=tf.float64), name='w_2', trainable=True)
#    self.bias_2 = tf.Variable(tf.truncated_normal([self.batch_size, 1], dtype=tf.float64), name='b_2', trainable=True)
#    self.hidden_2 = tf.add(tf.matmul(self.hidden_1, self.weights_2), self.bias_2)
#    self.hidden_2 = tf.nn.relu(self.hidden_2)
#    print('[hidden] {}'.format(self.hidden_2))
#
#    # hidden 3
#    self.weights_3 = tf.Variable(tf.truncated_normal([self.batch_size], dtype=tf.float64), name='w_3', trainable=True)
#    self.bias_3 = tf.Variable(tf.truncated_normal([self.batch_size, 1], dtype=tf.float64), name='b_3', trainable=True)
#    self.hidden_3 = tf.add(tf.matmul(self.hidden_1, self.weights_3), self.bias_3)
#    self.hidden_3 = tf.nn.relu(self.hidden_3)
#    print('[hidden] {}'.format(self.hidden_3))

    if self.mode == 'train':
      self.dropout = tf.nn.dropout(self.hidden_0, 0.3)

    self.global_step = self.global_step.assign_add(1)
    self.learning_rate = tf.train.exponential_decay(
        self.learning_rate,       # initial learning rate
        self.global_step,         # current index
        self.batch_size,          # decay step
        0.95,                     # decay rate
        staircase=True)

    self.logits = self.hidden_0
    self.pred = tf.nn.softmax(self.logits)
    self.loss = tf.losses.mean_squared_error(self.labels, self.pred)
    self.cost = tf.reduce_sum(tf.square(self.hidden_0 - self.labels))/(2 * self.batch_size)
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('learning_rate', self.learning_rate)
    self.saver = tf.train.Saver(tf.global_variables())

  def save(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      print('model saved @ {}'.format(self.saver.save(sess, self.model_path)))

  def load(self):
    if not gfile.Exists(self.model_path):
      return False
    with tf.Session() as sess:
      self.saver.restore(sess, self.model_path)
    return True

  def train(self):
    if not gfile.Exists(self.prep_data_path):
      preprocess(self.train_data_path)
    g = tf.get_default_graph()
    losses, costs = [], []
    with tf.Session(graph=g) as sess:
      tf.global_variables_initializer().run(session=sess)
      for e in range(self.total_epochs):
        loss = self.one_epoch(sess)
        print("epoch:{} loss:{}".format(sess.run(self.global_step), loss))
        print('chkpt saved @ {}'.format(self.saver.save(sess, self.chkpt_path, global_step=self.global_step)))

  def one_epoch(self, sess):
    sv = tf.train.Supervisor(logdir=self.logdir)
    with sv.managed_session(self.master) as sess:
      losses, costs = [], []
      for step, (x, y) in enumerate(generate_data(self.prep_data_path, self.batch_size)):
        if sv.should_stop():
          break
        sess.run([self.optimizer], feed_dict={
                self.feat_input: x,
                self.labels: y,
            })
        loss, cost, pred, lbls = sess.run([self.loss, self.cost, self.pred, self.labels],
            feed_dict={
                self.feat_input: x,
                self.labels: y,
            })

        print('step:{} lr:{} cost:{} loss:{}'.format(step, self.learning_rate.eval(), cost, loss))
        losses.append(loss)
        costs.append(cost)
        ax0.plot(np.arange(len(losses)), losses, '-', color='blue')
        ax0.plot(np.arange(len(costs)), costs, '+', color='r')
        ax0.legend(['loss', 'cost'], loc='upper right')
        ax1.plot(np.arange(len(pred)), pred, 'x', color='r')
        ax1.plot(np.arange(len(lbls)), lbls, 'o', color='g')
        ax1.legend(['pred', 'lbls'], loc='upper right')
        fig.canvas.draw()
        if self.global_step.eval() > self.steps_per_epoch:
          break
    return loss

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

def generate_data(source, chunksize=64, feat_nr=290, forever=True): 
  while True:
    reader = read_csv(source, header=0, chunksize=chunksize)
    for chunk in reader:
      x, y = load_chunk(chunk)
      x, y = np.resize(x, (chunksize, feat_nr)), np.resize(y, (chunksize, 1))
      x, y = x.astype(np.float64).tolist(), y.astype(np.float64).tolist()
      yield x, y
    if not forever: break

def plot_features(feat_input, labels, begin=0, end=50):
  with tf.Session() as sess:
    labels = sess.run(labels)
    feats = sess.run(feat_input)
  feats = np.reshape(feats, (feats.shape[1], feats.shape[0]))
  for x in feats[begin:end]:
      plt.plot(np.arange(len(x)), x)
  plt.plot(np.arange(len(labels)), labels, 'x', color='b')
  plt.show()

def main(_):
  global fig, ax0, ax1
  plt.ion()
  fig = plt.figure(figsize=(12, 8), facecolor='darkgray', edgecolor='black')
  ax0 = fig.add_subplot(211, facecolor='black')
  ax0.autoscale(True)
  ax1 = fig.add_subplot(212, facecolor='black')
  ax1.autoscale(True)
  #plt.title('profile')
  ax0.ylabel('metrics')
  ax0.xlabel('steps')
  ax1.ylabel('metrics')
  ax1.xlabel('steps')
  fig.show()
  args = init()
  graph = RusModel(args)
#  if not graph.load():
#  graph.create_model()
  graph.train()

if __name__ == '__main__':
  tf.app.run(main=main)

