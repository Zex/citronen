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
    self.global_epoch = 0#tf.Variable(1, trainable=False)
    self.global_step = tf.Variable(self.init_step, trainable=False)

  def build_model(self):
    # conv1
    self.X = tf.placeholder(shape=[self.batch_size, 1, 512, 660], dtype=tf.float32, name="x_input")
    self.target = tf.placeholder(shape=[self.batch_size], dtype=tf.int64, name="y")
    print(self.X)
    print(self.target)
    self.X = tf.nn.l2_normalize(self.X, 2)

    self.w_conv1 = tf.Variable(tf.truncated_normal([1, 1, 660, 512], stddev=0.215, dtype=tf.float32))
    self.conv1 = tf.nn.conv2d(self.X, self.w_conv1, [1, 1, 1, 1], padding='SAME', name='conv1')
    self.bias_conv1 = tf.Variable(tf.truncated_normal([512], stddev=0.215, dtype=tf.float32))
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
    self.w_conv3 = tf.Variable(tf.truncated_normal([1, 171, 220, 64], stddev=0.23, dtype=tf.float32))
    self.conv3 = tf.nn.conv2d(self.norm2, self.w_conv3, [1, 2, 2, 1], padding='SAME', name='conv3')
    self.bias_conv3 = tf.Variable(tf.truncated_normal([64], stddev=0.23, dtype=tf.float32))
    self.relu_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv3, self.bias_conv3))
    self.pool3 = tf.nn.max_pool(self.relu_conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
    self.norm3 = tf.nn.lrn(self.pool3, depth_radius=5, bias=1.0, name='norm3')
    print(self.norm3)
    # conv4
    self.w_conv4 = tf.Variable(tf.truncated_normal([1, 86, 64, 16], stddev=0.2, dtype=tf.float32))
    self.conv4 = tf.nn.conv2d(self.norm3, self.w_conv4, [1, 5, 5, 1], padding='SAME', name='conv4')
    self.bias_conv4 = tf.Variable(tf.truncated_normal([16], stddev=0.2, dtype=tf.float32))
    self.relu_conv4 = tf.nn.relu(tf.nn.bias_add(self.conv4, self.bias_conv4))
    self.pool4 = tf.nn.max_pool(self.relu_conv4, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')
    self.norm4 = tf.nn.lrn(self.pool4, depth_radius=5, bias=1.0, name='norm4')
    print(self.norm4)
    # conv5
    self.w_conv5 = tf.Variable(tf.truncated_normal([1, 18, 16, 2], stddev=0.015, dtype=tf.float32))
    self.conv5 = tf.nn.conv2d(self.norm4, self.w_conv5, strides=[1, 2, 2, 1], padding='SAME', name='conv5')
    self.bias_conv5 = tf.Variable(tf.truncated_normal([2], stddev=0.015, dtype=tf.float32))
    self.relu_conv5 = tf.nn.relu(tf.nn.bias_add(self.conv5, self.bias_conv5))
    self.pool5 = tf.nn.max_pool(self.relu_conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    self.norm5 = tf.nn.lrn(self.pool5, depth_radius=5, bias=1.0, name='norm5')
    print(self.norm5)
    
    # conv6
    self.w_conv6 = tf.Variable(tf.truncated_normal([1, 5, 2, 1], stddev=0.1, dtype=tf.float32))
    self.conv6 = tf.nn.conv2d(self.norm5, self.w_conv6, strides=[1, 2, 4, 1], padding='SAME', name='conv6')
    self.bias_conv6 = tf.Variable(tf.truncated_normal([1], stddev=0.1, dtype=tf.float32))
    self.relu_conv6 = tf.nn.relu(tf.nn.bias_add(self.conv6, self.bias_conv6))
    self.pool6 = tf.nn.max_pool(self.relu_conv6, ksize=[1, 1, 1, 1], strides=[1, 1, 2, 1], padding='SAME', name='pool6')
    self.norm6 = tf.nn.lrn(self.pool6, depth_radius=5, bias=1.0, name='norm6')
    print(self.norm6)
    self.output = tf.reshape(self.norm6, [1], name='output')

#   FC layers
#    self.flat = tf.reshape(self.norm5, [self.batch_size, 10])
#    print(self.flat)

#    self.fc1_w = tf.Variable(tf.truncated_normal([10, self.batch_size], stddev=0.1, dtype=tf.float32)) 
#    self.fc1_bias = tf.Variable(tf.truncated_normal([self.batch_size], stddev=0.1, dtype=tf.float32))
#    self.fc1_relu = tf.nn.relu(tf.add(tf.matmul(self.flat, self.fc1_w), self.fc1_bias))
#    self.fc1 = tf.nn.dropout(self.fc1_relu, 0.3, name='fc1')
#    print(self.fc1)
#
#    self.fc2_w = tf.Variable(tf.truncated_normal([self.batch_size, self.batch_size], stddev=0.1, dtype=tf.float32)) 
#    self.fc2_bias = tf.Variable(tf.truncated_normal([self.batch_size], stddev=0.1, dtype=tf.float32))
#    self.fc2_relu = tf.nn.relu(tf.add(tf.matmul(self.fc1, self.fc2_w), self.fc2_bias))
#    self.output = tf.nn.dropout(self.fc2_relu, 0.3, name='output')
    print(self.output)

    #self.pred = tf.argmax(self.output, 0, name='pred')
    self.pred = tf.reduce_mean(self.output, name='pred')
    print(self.pred)
    # calculate loss
    self.logits=tf.cast(tf.argmax(self.output, -1), tf.float32)
    self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
                labels=self.target,
                logits=self.output,#tf.squeeze(self.output),
                ), name='xentropy')
    print(self.loss)

    # apply gradient
    lr = tf.train.exponential_decay(self.lr, self.global_step, 100000, self.decay_rate, staircase=True)
    #self.opt = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=self.global_step)
    self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step)
    self.acc = tf.reduce_mean(tf.cast(
            tf.equal(self.output, tf.cast(self.target, tf.float32)), tf.float32), name='acc')
            #tf.equal(tf.cast(tf.argmax(self.output, 1), tf.float32), self.target), tf.float32), name='acc')
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
    last_loss = -0.1

    for i, (data, y) in enumerate(data_generator(self.data_root, self.label_path)):
      if data is None or np.isnan(y):
         continue

      data[np.where(data < 10000)] = 0.
      data = np.reshape(data, (1, 1, *data.shape))
      y = [y]

      if i % 5 == 0: # eval
        output, loss, acc, pred = self.sess.run([self.output, self.loss, self.acc, self.pred], feed_dict={
            self.X: data,
            self.target: y
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
        print(loss)
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

def conv():
  args = init()
  model = PassengerScreening(args)
  model.load()
  if model.mode == 'train':
    model.train()

def fit_data_generator(args):

    for x, y in data_generator(args.data_root, args.label_path):
      if x is None:# or not y.size():
         continue
      x[np.where(x < 10000)] = 0
      data = x.reshape(64, 512, 660)
      labels = y
      break

    X, y = tf.train.shuffle_batch(
        tensors=[data, labels],
        batch_size=64,
        capacity=1000,
        min_after_dequeue=300,
        )
    return {'image':X}, y

def eval_data_generator(args):

    for x, y in data_generator(args.data_root, args.label_path):
      if x is None:# or not y.size():
         continue
      x[np.where(x < 10000)] = 0
      data = x
      labels = y
      break

    X, y = tf.train.shuffle_batch(
        tensors=[data, labels],
        batch_size=args.batch_size,
        capacity=1000,
        min_after_dequeue=300,
        )
    return {'image':X}, y

def linear_reg():
  args = init()
  feats = tf.contrib.layers.real_valued_column('image', dimension=512*660)
  model = tf.contrib.learn.LinearRegressor(
    #example_id_column='Id',
    feature_columns=[feats],
    #n_classes=2,
    model_dir=args.model_root,
    #l2_regularization=0.11,
  )

  model.fit(
    input_fn=lambda: fit_data_generator(args),
    steps=1000)
  eval_res = model.evaluate(input_fn=lambda: eval_data_generator(args))
  print('eval res: {}'.format(eval_res))


def l2_norm(x):
    x_sqrt = np.sqrt(max(np.sum(x**2), 1e-12))
    return x / x_sqrt 
    
def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv_dense():
    args = init()
    w, h = 512, 660
    dim = w*h

    # Input
    X = tf.placeholder(shape=[args.batch_size, w, h, 1], dtype=tf.float32, name="x_input")
    target = tf.placeholder(shape=[args.batch_size, 2], dtype=tf.float32, name="y")
    print(X, flush=True)
    print(target, flush=True)

    # Conv1
    conv1 = tf.nn.relu(tf.nn.conv2d(X, weight_var([4, 4, 1, 32]), [1, 2, 2, 1], 
                padding='SAME', name='conv1') + bias_var([32]))
    norm1 = tf.nn.lrn(conv1, depth_radius=5, bias=1.0, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print(pool1, flush=True)

    # Conv2
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight_var([4, 4, 32, 64]), [1, 2, 2, 1],
                padding='SAME', name='conv2') + bias_var([64]))
    norm2 = tf.nn.lrn(conv2, depth_radius=5, bias=1.0, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print(pool2, flush=True)

    # Conv3
    conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight_var([3, 3, 64, 128]), [1, 2, 2, 1],
                padding='SAME', name='conv3') + bias_var([128]))
    norm3 = tf.nn.lrn(conv3, depth_radius=5, bias=1.0, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    print(pool3, flush=True)

    # Conv4
    conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weight_var([3, 3, 128, 512]), [1, 2, 2, 1],
                padding='SAME', name='conv4') + bias_var([512]))
    norm4 = tf.nn.lrn(conv4, depth_radius=5, bias=1.0, name='norm4')
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    print(pool4, flush=True)

    flat = tf.reshape(pool4, [-1, 2*3*512])

    # FC1
    fc1_w = weight_var([int(flat.get_shape()[1]), 1024])
    fc1_bias = bias_var([1024])
    fc1 = tf.nn.relu6(tf.add(tf.matmul(flat, fc1_w), fc1_bias), name='fc1')
    print(fc1)

    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu6, use_bias=True, name='dense1')
    print(dense1, flush=True) 

    dense2 = tf.layers.dense(inputs=dense1, units=2, activation=tf.nn.relu6, use_bias=True, name='dense2')
    print(dense2, flush=True) 

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=target,
        logits=dense2,
        #pos_weight=.1,
        name='loss'
        ))
    print(loss, flush=True)

    global_step = tf.Variable(args.init_epoch+1, trainable=False)
    global_epoch = args.init_epoch

    lr = tf.train.exponential_decay(
         learning_rate=args.lr,
         global_step=global_step,
         decay_steps=10000,
         decay_rate=args.decay_rate,
         staircase=True)
    opt = tf.train.AdamOptimizer(lr).minimize(
        loss,
        global_step=global_step)

    pred = tf.equal(tf.argmax(dense2, 0), tf.argmax(target, 0))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.image('X', tf.reshape(X, (args.batch_size, int(X.get_shape()[1]), int(X.get_shape()[2]), 1)))
    tf.summary.image('pool1', tf.reshape(pool1, (args.batch_size*32, int(pool1.get_shape()[1]), int(pool1.get_shape()[2]), 1)))
    tf.summary.image('pool2', tf.reshape(pool2, (args.batch_size*64, int(pool2.get_shape()[1]), int(pool2.get_shape()[2]), 1)))
    tf.summary.image('pool3', tf.reshape(pool3, (args.batch_size*128, int(pool3.get_shape()[1]), int(pool3.get_shape()[2]), 1)))
    tf.summary.image('pool4', tf.reshape(pool4, (args.batch_size*512, int(pool4.get_shape()[1]), int(pool4.get_shape()[2]), 1)))

    merged = tf.summary.merge_all()
    print(merged, flush=True)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      train_writer = tf.summary.FileWriter(args.model_root, sess.graph)

      for e in range(args.epochs):
        global_epoch = args.init_epoch + e + 1

        for i, (x, y) in enumerate(data_generator(args.data_root, args.label_path)):
          if x is None:
            continue
  
          filterd = []
          for d in range(args.batch_size):
            b = np.array(x[d,:])
            b[np.where(b < 12000)] = 0.
            b[np.where(b >= 12000)] = 255.
            filterd.append(b)
          filterd = np.array(filterd)

          #x = l2_norm(x)
          x = x.reshape(args.batch_size, w, h, 1).astype(np.float32)
 
          _, loss_val, acc_val, output, summary = sess.run([
              opt, loss, acc, dense2, merged], feed_dict={
                 X: x,
                 target: y
              })
          train_writer.add_summary(summary)

          print("[{}/{}]loss: {} acc: {} target: {} output: {} pred: {}".format(
              global_epoch,
              sess.run(global_step),
              loss_val,
              acc_val,
              np.mean(y),
              output,
              0.0
              ), flush=True)
        print('model saved @ {}'.format(saver.save(sess, args.model_path)), flush=True)
        

def start():
  conv_dense()

if __name__ == '__main__':
  start()


