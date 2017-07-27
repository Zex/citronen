from common import init, plot_img, init_axs, data_generator
from datetime import datetime
from pandas import read_csv, DataFrame
from os.path import isfile, basename
import matplotlib.gridspec as gridspec
import numpy as np
import inspect
import tensorflow as tf
from tensorflow import gfile


class PassengerScreening:

  def __init__(self, args):
    super(PassengerScreening, self).__init__()
    self.total_class = 2
    self.x_pts = 512 
    self.y_pts = 660
    self.rnn_size = 128
    self.mode = args.mode
    self.batch_size = args.batch_size
    self.epochs = args.epochs
    self.decay_rate = args.decay_rate
    self.lr = args.lr
    self.data_root = args.data_root
    self.label_path = args.label_path
    self.build()
    self.sess = tf.Session()
    self.model_path = '{}/{}.chkpt'.format(args.model_root, args.model_id)
    self.chkpt_path = args.chkpt
    self.summaries = []

  def loss_fn(self, output, target):
    return tf.reduce_mean(tf.abs(output-target), name='loss')

  def build(self):
#    self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
#    self.initial_state = self.lstm_cell.zero_state(self.batch_size)
    self.global_epoch = tf.Variable(1, trainable=False)
    #self.w = tf.get_variable('w', [self.rnn_size, self.x_pts, self.y_pts], tf.float32, tf.random_normal_initializer())
    #self.b = tf.get_variable('b', [self.rnn_size, self.x_pts, self.y_pts], tf.float32, tf.random_normal_initializer())
    self.batch_size = 1

  def svm_model(self):
    self.X = tf.placeholder(shape=[512, 660], dtype=tf.float32, name="x_input")
    self.target = tf.placeholder(shape=[1,], dtype=tf.float32, name="y")
    
    self.A = tf.Variable(tf.random_normal(shape=[660, 1]))
    self.b = tf.Variable(tf.random_normal(shape=[1, 1]))

    #self.output = tf.subtract(tf.matmul(self.X, self.A), self.b)
    #self.output = tf.reduce_mean(tf.nn.softmax(self.output))
    self.w_conv1 = tf.Variable(tf.truncated_normal([1, 512, 660, 1], stddev=0.1, dtype=tf.float32))
    self.conv1 = tf.nn.conv2d(self.X, self.w_conv1, [1, 3, 3, 1], padding='SAME', name='conv1')
    self.bias_conv1 = tf.Variable(tf.truncated_normal([512], stddev=0.1, dtype=tf.float32))
    self.relu_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.bias_conv1))
    self.pool1 = tf.nn.max_pool(self.relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
    self.norm1 = tf.nn.lrn(self.pool1, depth_radius=5, bias=1.0, name='norm1')
    print(self.norm1)
    
    # fc1
    self.w_fc1 = tf.Variable(tf.truncated_normal([1, 1, 10, 5], stddev=0.1, dtype=tf.float32))
    self.bias_fc1 = tf.Variable(tf.truncated_normal([5]), dtype=tf.float32)
    self.fc1 = tf.add(tf.matmul(self.norm1, self.w_fc1), bias_fc1, name='fc1')
    print(self.fc1)

    self.output = self.fc1

    self.l2_norm = tf.reduce_sum(tf.square(self.A), name='l2norm')
    self.alpha = tf.constant([0.012]) 
    self.classifier = tf.maximum(0., 1.-tf.multiply(self.output, self.target))
    self.loss = tf.add(self.classifier, tf.multiply(self.alpha, self.l2_norm), name='loss') 
    self.acc = tf.reduce_mean(tf.cast(tf.equal(self.output, self.target), tf.float32), name='acc')
    self.opt = tf.train.AdamOptimizer().minimize(self.loss)

  def svm_epoch(self):
    _loss, _acc = None, None
    for i, (data, y) in enumerate(data_generator(self.data_root, self.label_path)):
      if y.shape[0] == 0:
        continue
      y = y.astype(np.float32)
      # train
      pred = self.sess.run(self.output, feed_dict={
         self.X: data,
         self.target: y,
      })
      print('pred', pred, y, flush=True)
      self.sess.run(self.opt, feed_dict={
         self.X: data,
         self.target: y,
      })
      # calc loss
      _loss = self.sess.run(self.loss, feed_dict={
         self.X: data,
         self.target: y,
      })
      #_loss = self.sess.run(tf.reduce_mean(_loss))
      # calc acc
      _acc = self.sess.run(self.acc, feed_dict={
         self.X: data,
         self.target: y,
      })
      if i % 100 == 0:
        print("[{}] _A:{} _b:{} _loss:{} _acc:{}, _y:{}, [{}]".format(i,
              str(self.sess.run(self.A)), str(self.sess.run(self.b)),
              str(_loss), str(_acc), y, i), flush=True)
    return _loss, _acc

  def build_model(self):
    # conv1
    self.X = tf.placeholder(shape=[1, 1, 512, 660], dtype=tf.float32, name="x_input")
    self.target = tf.placeholder(shape=[1], dtype=tf.int32, name="y")

    self.w_conv1 = tf.Variable(tf.truncated_normal([1, 1, 660, 512], stddev=0.1, dtype=tf.float32))
    self.conv1 = tf.nn.conv2d(self.X, self.w_conv1, [1, 1, 1, 1], padding='SAME', name='conv1')
    self.bias_conv1 = tf.Variable(tf.truncated_normal([512], stddev=0.1, dtype=tf.float32))
    self.relu_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv1, self.bias_conv1))
    self.pool1 = tf.nn.max_pool(self.relu_conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
    self.norm1 = tf.nn.lrn(self.pool1, depth_radius=5, bias=1.0, name='norm1')
    print(self.norm1)
    # conv2
    self.w_conv2 = tf.Variable(tf.truncated_normal([1, 512, 512, 220], stddev=0.1, dtype=tf.float32))
    self.conv2 = tf.nn.conv2d(self.norm1, self.w_conv2, [1, 3, 3, 1], padding='SAME', name='conv2')
    self.bias_conv2 = tf.Variable(tf.truncated_normal([220], stddev=0.1, dtype=tf.float32))
    self.relu_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2, self.bias_conv2))
    self.pool2 = tf.nn.max_pool(self.relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
    self.norm2 = tf.nn.lrn(self.pool2, depth_radius=5, bias=1.0, name='norm2')
    print(self.norm2)
    # conv3
    self.w_conv3 = tf.Variable(tf.truncated_normal([1, 171, 220, 64], stddev=0.1, dtype=tf.float32))
    self.conv3 = tf.nn.conv2d(self.norm2, self.w_conv3, [1, 2, 2, 1], padding='SAME', name='conv3')
    self.bias_conv3 = tf.Variable(tf.truncated_normal([64], stddev=0.1, dtype=tf.float32))
    self.relu_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv3, self.bias_conv3))
    self.pool3 = tf.nn.max_pool(self.relu_conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
    self.norm3 = tf.nn.lrn(self.pool3, depth_radius=5, bias=1.0, name='norm3')
    print(self.norm3)
    # conv4
    self.w_conv4 = tf.Variable(tf.truncated_normal([1, 86, 64, 16], stddev=0.1, dtype=tf.float32))
    self.conv4 = tf.nn.conv2d(self.norm3, self.w_conv4, [1, 5, 5, 1], padding='SAME', name='conv4')
    self.bias_conv4 = tf.Variable(tf.truncated_normal([16], stddev=0.1, dtype=tf.float32))
    self.relu_conv4 = tf.nn.relu(tf.nn.bias_add(self.conv4, self.bias_conv4))
    self.pool4 = tf.nn.max_pool(self.relu_conv4, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')
    self.norm4 = tf.nn.lrn(self.pool4, depth_radius=5, bias=1.0, name='norm4')
    print(self.norm4)
    # conv5
    self.w_conv5 = tf.Variable(tf.truncated_normal([1, 18, 16, 2], stddev=0.1, dtype=tf.float32))
    self.conv5 = tf.nn.conv2d(self.norm4, self.w_conv5, strides=[1, 2, 2, 1], padding='SAME', name='conv5')
    self.bias_conv5 = tf.Variable(tf.truncated_normal([2], stddev=0.1, dtype=tf.float32))
    self.relu_conv5 = tf.nn.relu(tf.nn.bias_add(self.conv5, self.bias_conv5))
    self.pool5 = tf.nn.max_pool(self.relu_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    self.norm5 = tf.nn.lrn(self.pool5, depth_radius=5, bias=1.0, name='norm5')
    print(self.norm5)

    self.flat = tf.reshape(self.norm5, [5, 2])
    print(self.flat)

    self.fc1_w = tf.Variable(tf.truncated_normal([1, 5], stddev=0.1, dtype=tf.float32)) 
    self.fc1_bias = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float32))
    self.output = tf.add(tf.matmul(self.fc1_w, self.flat), self.fc1_bias)
    self.output = tf.nn.dropout(self.output, 0.5, name='output')
    #self.output = tf.reshape(self.norm5, [1, 2], name='output')#tf.nn.softmax(self.norm5)
    #self.output = tf.reduce_mean(self.output, name='output')
    print(self.output)
    # calculate loss
    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.output, labels=self.target), name='xentropy')
    print(self.loss)
    # apply gradient
    print('init param', self.lr, self.global_epoch, self.epochs, self.decay_rate)
    lr = tf.train.exponential_decay(self.lr, self.global_epoch, self.epochs, self.decay_rate)
    self.opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
    self.acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.argmax(self.output, 1), tf.int32), self.target), tf.float32))
    print(self.acc)

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.acc)

    self.saver = tf.train.Saver()

  def train(self):
    if True:
#      self.svm_model()
      self.build_model()
      self.sess.run(tf.global_variables_initializer())
      losses, accs = [], []
      for i in range(self.epochs):
#        _loss, _acc = self.svm_epoch()
        _loss, _acc = self.epoch()
        losses.append(_loss)
        accs.append(_acc)
      self.save(self.sess)

  def epoch(self):
    for i, (data, y) in enumerate(data_generator(self.data_root, self.label_path)):
      if y.shape[0] == 0:
         continue
      data = np.reshape(data, (1, 1, 512, 660))
      output, _, loss, acc = self.sess.run([self.output, self.opt, self.loss, self.acc], feed_dict={
        self.X: data,
        self.target: y
      })
      #loss, acc = self.step(output, y)
      print("[{}] loss: {} acc: {} target: {}".format(
        self.sess.run(self.global_epoch), loss, acc, y), flush=True)
    return loss, acc

  def save(self):
    self.saver = tf.train.Saver(tf.global_variables())
    print('model saved @ {}'.format(self.saver.save(self.sess, self.model_path)))

  def load(self):
    if not gfile.Exists(self.model_path):
      return False
    with tf.Session() as sess:
      self.saver.restore(sess, self.model_path)
    return True

def start():
  args = init()
  model = PassengerScreening(args)
  model.load()
  if model.mode == 'train':
    model.train()


if __name__ == '__main__':
  start()

