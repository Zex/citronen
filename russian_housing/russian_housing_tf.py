
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

class RusTf:
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
    #self.create_model()

  def update_path(self):
    self.model_path = "../models/{}.h5".format(self.model_id)
    self.chkpt_path = "../models/{}.chkpt".format(self.model_id)
    self.logdir = "../build/{}.log".format(self.model_id)
    self.prep_data_path = '../build/{}-prep.csv'.format(self.model_id)
    self.data_base = '../data/russian_housing'
    self.train_data_path = "{}/train.csv".format(self.data_base)
    self.eval_data_path = "{}/eval.csv".format(self.data_base)
    self.test_data_path = "{}/test.csv".format(self.data_base)

  def input_fn(self, source):
    feature_columns = self.get_feature_columns(self.train_data_path)
  
    data = read_csv(source, header=0, nrows=self.batch_size)
    cols = [tf.contrib.layers.real_valued_column(k) for k in self.feature_columns]
    x = data.iloc[0:data.shape[0], 1:291]
    x = {k: tf.constant(x[k].values) for k in feature_columns}
    y = tf.constant(data['price_doc'].values)
    return x, y 

  def regression(self, feature_columns):
    self.feature_columns = [tf.contrib.layers.real_valued_column(k) for k in feature_columns]
    self.regressor = tf.contrib.learn.DNNRegressor(
            feature_columns=self.feature_columns, hidden_units=[500, 100, 64])
    self.regressor.fit(input_fn=lambda: self.input_fn(self.train_data_path), steps=self.steps_per_epoch)
    for step, (x, y) in enumerate(generate_data(self.prep_data_path, self.batch_size)):
      x = np.squeeze(x)
      pred = list(self.regressor.predict(x, as_iterable=True))
      print('pred:{}'.format(pred))
      score = tf.losses.mean_squared_error(y, pred)
      print('score:{}'.format(score))

  def create_model(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    self.feat_input = tf.placeholder(tf.float64, shape=(self.batch_size, self.max_features), name='feat_input')
    self.labels = tf.placeholder(tf.float64, shape=(self.batch_size, 1), name='labels')
    self.feat_input = tf.truncated_normal(self.feat_input.shape, dtype=tf.float64)
    self.labels = tf.truncated_normal(self.labels.shape, dtype=tf.float64)
    self.feat_input = tf.reshape(self.feat_input, [1, 64, 290]) 
    print('[feat_input] {} [labels] {}'.format(self.feat_input, self.labels))
    # conv0
    with tf.variable_scope('conv0') as scope:
      with tf.device('/cpu:0'):
        kernel = tf.get_variable('kernel', [1, 290, 64], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        bias = tf.get_variable('bias', [1, 64, 64], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv1d(self.feat_input, kernel, 1, padding='SAME')
        conv = tf.nn.relu(tf.add(conv, bias), name=scope.name)
        print('[conv] {}'.format(conv))

    # max pool 0
    conv = tf.to_float(conv)
    conv = tf.reshape(conv, [1, 1, 64, 64])
    pool0 = tf.nn.max_pool(conv, [1, 1, 2, 2], [1, 1, 1, 1], padding='SAME', name='pool_0')
    print('[pool] {}'.format(pool0))

    # conv1
    with tf.variable_scope('conv1') as scope:
      with tf.device('/cpu:0'):
        kernel = tf.get_variable('kernel', [1, 290, 64], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        bias = tf.get_variable('bias', [1, 64, 64], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv1d(self.feat_input, kernel, 1, padding='SAME')
        conv = tf.nn.relu(tf.add(conv, bias), name=scope.name)
        print('[conv] {}'.format(conv))

    # max pool 1
    conv = tf.to_float(conv)
    conv = tf.reshape(conv, [1, 1, 64, 64])
    pool1 = tf.nn.max_pool(conv, [1, 1, 2, 2], [1, 1, 1, 1], padding='SAME', name='pool_1')
    print('[pool] {}'.format(pool1))

    pool1 = tf.squeeze(pool1)
    pool1 = tf.to_double(pool1)
    print('[pool] {}'.format(pool1))

    # hidden 0
    with tf.variable_scope('hidden0') as scope:
#      self.weights_0 = tf.Variable(tf.truncated_normal([self.feat_input.get_shape()[1].value, 580], dtype=tf.float64, stddev=0.05), name='w_0', trainable=True)
#      self.bias_0 = tf.Variable(tf.truncated_normal([self.batch_size, 1], dtype=tf.float64), name='b_3', trainable=True)
      with tf.device('/cpu:0'):
        self.weights_0 = tf.get_variable('weights', [64, 580], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        self.bias_0 = tf.get_variable('biases', [580], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
      self.hidden_0 = tf.nn.relu(tf.add(tf.matmul(pool1, self.weights_0), self.bias_0), name=scope.name)
      print('[hidden] {}'.format(self.hidden_0))
      if self.mode == 'train':
        self.hidden_0 = tf.nn.dropout(self.hidden_0, 0.2)

    # hidden 1
    with tf.variable_scope('hidden1') as scope:
      with tf.device('/cpu:0'):
        self.weights_1 = tf.get_variable('weights', [580, 64], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        self.bias_1 = tf.get_variable('biases', [64], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
      self.hidden_1 = tf.nn.relu(tf.add(tf.matmul(self.hidden_0, self.weights_1), self.bias_1), name=scope.name)
      print('[hidden] {}'.format(self.hidden_1))
      if self.mode == 'train':
        self.hidden_1 = tf.nn.dropout(self.hidden_1, 0.3)

    # hidden 2
    with tf.variable_scope('hidden2') as scope:
      with tf.device('/cpu:0'):
        self.weights_2 = tf.get_variable('weights', [64, 128], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        self.bias_2 = tf.get_variable('biases', [128], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
      self.hidden_2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_1, self.weights_2), self.bias_2), name=scope.name)
      print('[hidden] {}'.format(self.hidden_2))
      if self.mode == 'train':
        self.hidden_2 = tf.nn.dropout(self.hidden_2, 0.2)

    # hidden 3
    with tf.variable_scope('hidden3') as scope:
      with tf.device('/cpu:0'):
        self.weights_3 = tf.get_variable('weights', [64, 1], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        self.bias_3 = tf.get_variable('biases', [64], dtype=tf.float64, initializer=tf.constant_initializer(0.1))
      self.hidden_3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_1, self.weights_3), self.bias_3), name=scope.name)
      print('[hidden] {}'.format(self.hidden_3))
      if self.mode == 'train':
        self.hidden_3 = tf.nn.dropout(self.hidden_3, 0.2)

#    cell_fw = tf.nn.rnn_cell.LSTMCell(128, 
#          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=512))
#    cell_bw = tf.nn.rnn_cell.LSTMCell(128, 
#          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=128))
#    outputs, output_state_fw, output_state_bw = tf.contrib.rnn.bidirectional_rnn(
#      cell_fw, cell_bw, self.x_input, dtype=tf.float64)

    self.learning_rate = tf.train.exponential_decay(
        self.learning_rate,       # initial learning rate
        tf.contrib.framework.get_global_step(),         # current index
        self.batch_size,          # decay step
        0.95,                     # decay rate
        staircase=True)

    with tf.variable_scope('logits') as scope:
      with tf.device('/cpu:0'):
        self.weights_logits = tf.get_variable('weights', [64, 1], dtype=tf.float64, initializer=tf.truncated_normal_initializer(0.05))
        self.bias_logits = tf.get_variable('biases', [1], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
      self.logits = tf.add(tf.matmul(self.hidden_3, self.weights_logits), self.bias_logits, name=scope.name)
      print('[logits] {}'.format(self.logits))

    self.pred = tf.nn.softmax(self.logits)
    self.loss = tf.losses.mean_squared_error(self.labels, self.pred)
    self.cost = tf.reduce_sum(tf.square(self.logits - self.labels))/(2 * self.batch_size)
    self.total_loss = tf.losses.get_total_loss()
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('learning_rate', self.learning_rate)
    print('='*80)
    [print(str(v)) for v in tf.global_variables()]
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

  def get_feature_columns(self, source):
    data = read_csv(source, header=0, nrows=1)
    return data.columns.values[1:291].tolist()

  def train(self):
    if not gfile.Exists(self.prep_data_path):
      preprocess(self.train_data_path)
    g = tf.get_default_graph()
    self.regression(self.get_feature_columns(self.train_data_path))
    return
    losses, costs = [], []
    with tf.Session(graph=g) as sess:
      tf.global_variables_initializer().run(session=sess)
      for e in range(self.total_epochs):
        loss = self.one_epoch(sess)
        global_step = tf.contrib.framework.get_global_step()
        print("epoch:{} loss:{}".format(global_step, loss))
        print('chkpt saved @ {}'.format(self.saver.save(sess, self.chkpt_path, global_step=global_step)))

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
        loss, cost, pred, lbls = sess.run([self.total_loss, self.cost, self.pred, self.labels],
            feed_dict={
                self.feat_input: x,
                self.labels: y,
            })
        h0, h1 = sess.run([self.hidden_0, self.hidden_1],
            feed_dict={
                self.feat_input: x,
                self.labels: y,
            })
        print("h0:{}, h1:{}".format(h0, h1))
        #print('step:{} lr:{} cost:{} loss:{}'.format(step, self.learning_rate.eval(), cost, loss))
        """ 
        losses.append(loss)
        costs.append(cost)
        ax0.plot(np.arange(len(losses)), losses, '-', color='blue')
        ax0.plot(np.arange(len(costs)), costs, '+', color='r')
        ax0.legend(['loss', 'cost'], loc='upper right')
        ax1.plot(np.arange(len(pred)), pred, 'x', color='r')
        ax1.plot(np.arange(len(lbls)), lbls, 'o', color='g')
        ax1.legend(['pred', 'lbls'], loc='upper right')
        fig.canvas.draw()
        """
        #if self.global_step.eval() > self.steps_per_epoch:
        if step > self.steps_per_epoch: 
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

  def is_train(self):
    return self.mode == 'train'

  def is_eval(self):
    return self.mode == 'eval'

  def is_test(self):
    return self.mode == 'test'

  def generate_train_data(self):
    yield generate_data(self.train_data_path, self.batch_size, self.max_features)


def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=RusTf.modes)
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
  x, y = x.astype(np.float64), y.astype(np.float64)
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

def setup_plot():
  global fig, ax0, ax1
  plt.ion()
  fig = plt.figure(figsize=(8, 6), facecolor='darkgray', edgecolor='black')
  ax0 = fig.add_subplot(211, facecolor='black')
  ax0.autoscale(True)
  ax1 = fig.add_subplot(212, facecolor='black')
  ax1.autoscale(True)
  #fig.show()

def main(_):
  args = init()
  model = RusTf(args)
#  setup_plot()

  if model.is_train():
    model.train()
  elif rus.is_eval():
    model.eval()
  else:
    model.test()

if __name__ == '__main__':
  tf.app.run(main=main)

