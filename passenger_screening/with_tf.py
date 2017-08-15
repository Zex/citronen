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
    

def conv_dense():
    args = init()
    w, h = 512, 660
    dim = w*h

    # Input
    X = tf.placeholder(shape=[1, h, w, args.batch_size], dtype=tf.float32, name="x_input")
    target = tf.placeholder(shape=[args.batch_size,], dtype=tf.float32, name="y")
    print(X)
    print(target)

    # Conv1
    w_conv1 = tf.Variable(tf.truncated_normal([w, h, args.batch_size, 32], stddev=0.1, dtype=tf.float32))
    conv1 = tf.nn.conv2d(X, w_conv1, [1, 2, 2, 1], padding='SAME', name='conv1')
    bias_conv1 = tf.Variable(tf.truncated_normal([32], stddev=0.1, dtype=tf.float32))
    relu_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bias_conv1))
    norm1 = tf.nn.lrn(relu_conv1, depth_radius=5, bias=1.0, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool1')
    print(pool1)

    # Conv2
    w_conv2 = tf.Variable(tf.truncated_normal([256, 330, 32, 128], stddev=0.1, dtype=tf.float32))
    conv2 = tf.nn.conv2d(pool1, w_conv2, [1, 2, 2, 1], padding='SAME', name='conv2')
    bias_conv2 = tf.Variable(tf.truncated_normal([128], stddev=0.1, dtype=tf.float32))
    relu_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bias_conv2))
    norm2 = tf.nn.lrn(relu_conv2, depth_radius=5, bias=1.0, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool2')
    print(pool2)

    # Conv3
    w_conv3 = tf.Variable(tf.truncated_normal([128, 165, 128, 64], stddev=0.1, dtype=tf.float32))
    conv3 = tf.nn.conv2d(pool2, w_conv3, [1, 3, 3, 1], padding='SAME', name='conv3')
    bias_conv3 = tf.Variable(tf.truncated_normal([64], stddev=0.1, dtype=tf.float32))
    relu_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, bias_conv3))
    norm3 = tf.nn.lrn(relu_conv3, depth_radius=5, bias=1.0, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
    print(pool3)

    # Conv4
    w_conv4 = tf.Variable(tf.truncated_normal([55, 165, 64, 16], stddev=0.1, dtype=tf.float32))
    conv4 = tf.nn.conv2d(pool3, w_conv4, [1, 2, 2, 1], padding='SAME', name='conv4')
    bias_conv4 = tf.Variable(tf.truncated_normal([16], stddev=0.1, dtype=tf.float32))
    relu_conv4 = tf.nn.relu(tf.nn.bias_add(conv4, bias_conv4))
    norm4 = tf.nn.lrn(relu_conv4, depth_radius=5, bias=1.0, name='norm4')
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool4')
    print(pool4)

    # FC1
#    fc1_w = tf.Variable(tf.truncated_normal([dim, args.batch_size], stddev=0.1, dtype=tf.float32)) 
#    fc1_bias = tf.Variable(tf.truncated_normal([args.batch_size], stddev=0.1, dtype=tf.float32))
#    fc1 = tf.nn.relu(tf.add(tf.matmul(X, fc1_w), fc1_bias), name='fc1')
#    print(fc1)

    flat = tf.reshape(pool4, [64, 154])

    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu, use_bias=True, name='dense1')
    print(dense1) 

    dense2 = tf.layers.dense(inputs=flat, units=1, activation=tf.nn.relu, use_bias=True, name='dense2')
    print(dense2) 

#    nce_w = tf.Variable(tf.truncated_normal([2, args.batch_size], -1.0, 1.0))
#    nce_b = tf.Variable(tf.zeros([args.batch_size]))
#    loss = tf.nn.nce_loss(
#        weights=nce_w,#np.tile(0.5, [2, args.batch_size]).shape,
#        biases=nce_b,#[0.5]*args.batch_size,
#        labels=target,
#        inputs=fc1,
#        num_sampled=1,
#        num_classes=2,
#        num_true=args.batch_size,
#    )
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=target,
        logits=tf.squeeze(dense2),
        name='loss'
        ))
    print(loss)

    global_step = tf.Variable(args.init_epoch, trainable=False)
    global_epoch = args.init_epoch

    lr = tf.train.exponential_decay(
         learning_rate=args.lr,
         global_step=global_step,
         decay_steps=1000,
         decay_rate=args.decay_rate,
         staircase=True)
    opt = tf.train.AdamOptimizer(lr).minimize(
        loss,
        global_step=global_step)

    pred = tf.equal(tf.argmax(dense2, 0), tf.argmax(target, 0))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.image('pool1', tf.reshape(pool1, [330, 256, 32, 1]))
    tf.summary.image('pool2', tf.reshape(pool2, [165, 128, 128, 1]))
    tf.summary.image('pool3', tf.reshape(pool3, [55, 43, 64, 1]))
    tf.summary.image('pool4', tf.reshape(pool4, [28, 22, 16, 1]))

    merged = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      train_writer = tf.summary.FileWriter(args.model_root, sess.graph)

      for e in range(args.epochs):
        global_epoch = args.init_epoch + e + 1

        for i, (x, y) in enumerate(data_generator(args.data_root, args.label_path)):
          if x is None:
            continue
  
          #x[np.where(x < 10000)] = 0.
          y = y.astype(np.float32)
          x = l2_norm(x)
          x = x.reshape(1, h, w, x.shape[0]).astype(np.float32)
 
          _, loss_val, acc_val, summary = sess.run([opt, loss, acc, merged], feed_dict={
              X: x,
              target: y
              })
          train_writer.add_summary(summary)

          p1, p2, p3, p4, d1, d2 = sess.run([pool1, pool2, pool3, pool4, dense1, dense2])
          
  
          print("[{}/{}]loss: {} acc: {} target: {} output: {} pred: {}".format(
              global_epoch,
              sess.run(global_step),
              loss_val,
              acc_val,
              np.mean(y),
              0.0,
              0.0
              ), flush=True)
        print('model saved @ {}'.format(saver.save(sess, args.model_path)), flush=True)
        

def start():
  conv_dense()

if __name__ == '__main__':
  start()


