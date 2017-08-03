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
    self.init_epoch = args.init_epoch
    self.batch_size = args.batch_size
    self.epochs = args.epochs
    self.decay_rate = args.decay_rate
    self.lr = args.lr
    self.data_root = args.data_root
    self.label_path = args.label_path
    self.init_step = args.init_step
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
    self.global_epoch = 0#tf.Variable(1, trainable=False)
    self.global_step = tf.Variable(self.init_step, trainable=False)
    #self.w = tf.get_variable('w', [self.rnn_size, self.x_pts, self.y_pts], tf.float32, tf.random_normal_initializer())
    #self.b = tf.get_variable('b', [self.rnn_size, self.x_pts, self.y_pts], tf.float32, tf.random_normal_initializer())

  def build_model(self):
    # conv1
    self.X = tf.placeholder(shape=[self.batch_size, 1, 512, 660], dtype=tf.float32, name="x_input")
    self.target = tf.placeholder(shape=[self.batch_size], dtype=tf.float32, name="y")
    self.X = tf.nn.l2_normalize(self.X, 2)#, depth_radius=5, bias=1.0, name='nor_x')

    self.w_conv1 = tf.Variable(tf.truncated_normal([1, 1, 660, 512], stddev=0.215, dtype=tf.float32))
    self.conv1 = tf.nn.conv2d(self.X, self.w_conv1, [1, 1, 1, 1], padding='SAME', name='conv1')
    self.bias_conv1 = tf.Variable(tf.truncated_normal([512], stddev=0.215, dtype=tf.float32))
    self.relu_conv1 = tf.nn.relu6(tf.nn.bias_add(self.conv1, self.bias_conv1))
    self.norm1 = tf.nn.lrn(self.relu_conv1, depth_radius=5, bias=1.0, name='norm1')
    self.pool1 = tf.nn.max_pool(self.norm1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
    print(self.pool1)
    # conv2
    self.w_conv2 = tf.Variable(tf.truncated_normal([1, 512, 512, 220], stddev=0.1, dtype=tf.float32))
    self.conv2 = tf.nn.conv2d(self.pool1, self.w_conv2, [1, 3, 3, 1], padding='SAME', name='conv2')
    self.bias_conv2 = tf.Variable(tf.truncated_normal([220], stddev=0.1, dtype=tf.float32))
    self.relu_conv2 = tf.nn.relu6(tf.nn.bias_add(self.conv2, self.bias_conv2))
    self.norm2 = tf.nn.lrn(self.relu_conv2, depth_radius=5, bias=1.0, name='norm2')
    self.pool2 = tf.nn.max_pool(self.norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
    print(self.pool2)
    # conv3
    self.w_conv3 = tf.Variable(tf.truncated_normal([1, 171, 220, 64], stddev=0.23, dtype=tf.float32))
    self.conv3 = tf.nn.conv2d(self.pool2, self.w_conv3, [1, 2, 2, 1], padding='SAME', name='conv3')
    self.bias_conv3 = tf.Variable(tf.truncated_normal([64], stddev=0.23, dtype=tf.float32))
    self.relu_conv3 = tf.nn.relu6(tf.nn.bias_add(self.conv3, self.bias_conv3))
    self.pool3 = tf.nn.max_pool(self.relu_conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
    self.norm3 = tf.nn.lrn(self.pool3, depth_radius=5, bias=1.0, name='norm3')
    print(self.norm3)
    # conv4
    self.w_conv4 = tf.Variable(tf.truncated_normal([1, 86, 64, 16], stddev=0.2, dtype=tf.float32))
    self.conv4 = tf.nn.conv2d(self.pool3, self.w_conv4, [1, 5, 5, 1], padding='SAME', name='conv4')
    self.bias_conv4 = tf.Variable(tf.truncated_normal([16], stddev=0.2, dtype=tf.float32))
    self.relu_conv4 = tf.nn.relu6(tf.nn.bias_add(self.conv4, self.bias_conv4))
    self.norm4 = tf.nn.lrn(self.relu_conv4, depth_radius=5, bias=1.0, name='norm4')
    self.pool4 = tf.nn.max_pool(self.norm4, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')
    print(self.pool4)
    # conv5
    self.w_conv5 = tf.Variable(tf.truncated_normal([1, 18, 16, 2], stddev=0.015, dtype=tf.float32))
    self.conv5 = tf.nn.conv2d(self.pool4, self.w_conv5, strides=[1, 2, 2, 1], padding='SAME', name='conv5')
    self.bias_conv5 = tf.Variable(tf.truncated_normal([2], stddev=0.015, dtype=tf.float32))
    self.relu_conv5 = tf.nn.relu6(tf.nn.bias_add(self.conv5, self.bias_conv5))
    self.norm5 = tf.nn.lrn(self.relu_conv5, depth_radius=5, bias=1.0, name='norm5')
    self.pool5 = tf.nn.max_pool(self.norm5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    print(self.pool5)
    
    # conv6
    self.w_conv6 = tf.Variable(tf.truncated_normal([1, 5, 2, 1], stddev=0.1, dtype=tf.float32))
    self.conv6 = tf.nn.conv2d(self.pool5, self.w_conv6, strides=[1, 2, 4, 1], padding='SAME', name='conv6')
    self.bias_conv6 = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float32))
    self.relu_conv6 = tf.nn.relu6(tf.nn.bias_add(self.conv6, self.bias_conv6))
    self.norm6 = tf.nn.lrn(self.relu_conv6, depth_radius=5, bias=1.0, name='norm6')
    self.pool6 = tf.nn.max_pool(self.norm6, ksize=[1, 1, 1, 1], strides=[1, 1, 2, 1], padding='VALID', name='pool6')
    print(self.pool6)
#    self.flat = tf.reshape(self.pool5, [5, 2])
#    print(self.flat)

#    self.fc1_w = tf.Variable(tf.truncated_normal([1, 5], stddev=0.1, dtype=tf.float32)) 
#    self.fc1_bias = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float32))
#    self.output = tf.nn.relu6(tf.add(tf.matmul(self.fc1_w, self.flat), self.fc1_bias))
#    self.output = tf.nn.dropout(self.output, 0.3, name='output')
    self.output = tf.reshape(self.pool6, [self.batch_size, 1], name='output')
    print(self.output)

    self.pred = tf.argmax(self.output, 0, name='pred')
    print(self.pred)
    # calculate loss
    self.logits=tf.cast(tf.argmax(self.output, -1), tf.float32)
    self.loss = tf.reduce_mean(
        #tf.losses.softmax_cross_entropy(
        #tf.nn.sparse_softmax_cross_entropy_with_logits(
        #tf.nn.weighted_cross_entropy_with_logits(
        tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(self.output),
                labels=self.target,
                ), name='xentropy')
    print(self.loss)

    # apply gradient
    lr = tf.train.exponential_decay(self.lr, self.global_step, 100000, self.decay_rate, staircase=True)
    #self.opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)
    self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step)
    self.acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.argmax(self.output, 1), tf.float32), self.target), tf.float32), name='acc')
    print(self.acc)

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.acc)

    self.saver = tf.train.Saver()

  def train(self):
    self.build_model()
    self.sess.run(tf.global_variables_initializer())
    losses, accs = [], []
    for i in range(self.epochs):
      self.global_epoch = self.init_epoch + i + 1
      _loss, _acc = self.epoch()
      if np.isnan(_loss):
        break
      losses.append(_loss)
      accs.append(_acc)
    self.save()

  def epoch(self):
    total_acc_eval = []
    total_acc_train = []
    last_loss = None

    for i, (data, y) in enumerate(data_generator(self.data_root, self.label_path)):

      if data is None or len(y) == 0:
         continue

      data[np.where(data < 10000)] = 0.
      data = np.reshape(data, (self.batch_size, 1, 512, 660))

      if i % 5 == 0: # eval
        output, loss, acc, pred = self.sess.run([self.output, self.loss, self.acc, self.pred], feed_dict={
            self.X: data,
            self.target: np.array(y)
        })
        if np.isnan(loss):
          break
        total_acc_eval.append(acc)
        print("[{}/{}][eval] loss: {} acc: {} target: {} output: {} pred: {}".format(
          self.global_epoch, self.sess.run(self.global_step),
          loss,
          acc, #len([a for a in total_acc_eval if a])/len(total_acc_eval),
          y, 
          output,
          pred,
          ), flush=True)
      else: # train
        output, _, loss, acc = \
            self.sess.run([self.output, self.opt, self.loss, self.acc], feed_dict={
                self.X: data,
                self.target: y
        })
        if np.isnan(loss) or last_loss == loss:
          break
        total_acc_train.append(acc)
        print("[{}/{}] loss: {} acc: {} target: {} output: {}".format(
          self.global_epoch, self.sess.run(self.global_step),
          loss,
          acc, #len([a for a in total_acc_train if a])/len(total_acc_train),
          y, 
          output), flush=True)
      lass_loss = loss
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

