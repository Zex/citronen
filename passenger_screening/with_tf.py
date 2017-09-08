from common import init, plot_img, init_axs, data_generator
from datetime import datetime
from pandas import read_csv, DataFrame
from os.path import isfile, basename
import matplotlib.gridspec as gridspec
import numpy as np
import inspect
import tensorflow as tf
from tensorflow import gfile


def l2_norm(x):
    x_sqrt = np.sqrt(max(np.sum(x**2), 1e-12))
    return x / x_sqrt 
    
def weight_var(shape):
    #return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def bias_var(shape):
    #return tf.Variable(tf.constant(0.1, shape=shape))
    return tf.Variable(tf.random_normal(shape=shape))

def onehot(label, class_nr):
    buf = np.zeros((class_nr))
    buf[label] = 1
    return buf

def conv_dense():
    args = init()
    w, h = 512, 660

    # Input
    X = tf.placeholder(shape=[args.batch_size, w, h, 1], dtype=tf.float32, name="x_input")
    target = tf.placeholder(shape=[args.batch_size, 2], dtype=tf.float32, name="y")
    print(X, flush=True)
    print(target, flush=True)

    # Conv1
    conv1 = tf.nn.relu(tf.nn.conv2d(X, weight_var([4, 4, 1, 512]), [1, 2, 2, 1], 
                padding='SAME', name='conv1') + bias_var([512]))
    norm1 = tf.nn.lrn(conv1, depth_radius=5, bias=1.0, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print(conv1, pool1, flush=True)

    # Conv2
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weight_var([4, 4, 512, 128]), [1, 2, 2, 1],
                padding='SAME', name='conv2') + bias_var([128]))
    norm2 = tf.nn.lrn(conv2, depth_radius=5, bias=1.0, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print(conv2, pool2, flush=True)

    # Conv3
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weight_var([3, 3, 128, 32]), [1, 2, 2, 1],
                padding='SAME', name='conv3') + bias_var([32]))
    norm3 = tf.nn.lrn(conv3, depth_radius=5, bias=1.0, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    print(conv3, pool3, flush=True)

    # Conv4
    conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weight_var([3, 3, 32, 8]), [1, 2, 2, 1],
                padding='SAME', name='conv4') + bias_var([8]))
    norm4 = tf.nn.lrn(conv4, depth_radius=5, bias=1.0, name='norm4')
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    print(conv4, pool4, flush=True)

    #flat = tf.reshape(pool4, [-1, 2*3*512])
    flat = tf.reshape(conv4, [args.batch_size, 32*42*8])

#    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.elu, use_bias=True, name='dense1')
#    print(dense1, flush=True) 

    dense2 = tf.layers.dense(inputs=flat, units=2, activation=tf.nn.relu, use_bias=True, name='dense2')
    print(dense2, flush=True) 

    pred = tf.cast(tf.equal(dense2, target), tf.float32)
    acc = tf.reduce_sum(pred)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=target,
        logits=dense2,
        #pos_weight=.1,
        ), name='loss')
    grad = tf.gradients(loss, X)
    print('loss', loss)
    print('acc', acc)
    print('grad', grad)

    global_step = tf.Variable(args.init_epoch, trainable=False)
    global_epoch = args.init_epoch

    lr = tf.train.exponential_decay(
         learning_rate=args.lr,
         global_step=global_step,
         decay_steps=1e+5,
         decay_rate=args.decay_rate,
         staircase=True)
    opt = tf.train.AdamOptimizer(lr).minimize(
        loss,
        global_step=global_step)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.scalar('lr', lr)
    tf.summary.histogram('grad', grad)
    tf.summary.histogram('pred', pred)
    tf.summary.histogram('target', target)
    tf.summary.image('X', tf.reshape(X, (args.batch_size, int(X.get_shape()[1]), int(X.get_shape()[2]), 1)))

    tf.summary.image('conv1', tf.reshape(conv1, (
            int(conv1.get_shape()[0])*int(conv1.get_shape()[3]),
            int(conv1.get_shape()[1]), int(conv1.get_shape()[2]), 1)))
    tf.summary.image('conv2', tf.reshape(conv2, (
            int(conv2.get_shape()[0])*int(conv2.get_shape()[3]),
            int(conv2.get_shape()[1]), int(conv2.get_shape()[2]), 1)))
    tf.summary.image('conv3', tf.reshape(conv3, (
            int(conv3.get_shape()[0])*int(conv3.get_shape()[3]),
            int(conv3.get_shape()[1]), int(conv3.get_shape()[2]), 1)))
    tf.summary.image('conv4', tf.reshape(conv4, (
            int(conv4.get_shape()[0])*int(conv4.get_shape()[3]),
            int(conv4.get_shape()[1]), int(conv4.get_shape()[2]), 1)))

    for v in tf.trainable_variables():
        print(v.name, v.get_shape())
        if len(v.get_shape()) == 4:
          tf.summary.image(v.name, tf.reshape(v, (int(v.get_shape()[3])*int(v.get_shape()[2]), int(v.get_shape()[0]), int(v.get_shape()[1]), 1)))
        tf.summary.histogram(v.name, v)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      train_writer = tf.summary.FileWriter(args.model_root, sess.graph)

      for e in range(args.epochs):
        global_epoch = args.init_epoch + e + 1
        batch, labels = [], []
        for i, (x, y) in enumerate(data_generator(args.data_root, args.label_path)):
          if x is None:
            continue
  
          #x = x.reshape(args.batch_size, w, h, 1)
          #y = onehot(y, 2)
          if len(batch) < args.batch_size:
              batch.append(x)
              labels.append(onehot(y, 2))
          else:
            x = np.array(batch).reshape(args.batch_size, w, h, 1)
            y = np.array(labels)
            _, loss_val, acc_val, pred_val, summary = sess.run([
                opt, loss, acc, pred, merged], feed_dict={
                   X: x,
                   target: y
                })
            train_writer.add_summary(summary, sess.run(global_step))

            print("[{}/{}]loss: {} acc: {} target: {} pred: {} output: {}".format(
              global_epoch,
              sess.run(global_step),
              loss_val,
              acc_val,
              y,
              pred_val,
              -0.,
              ), flush=True)
        print('model saved @ {}'.format(saver.save(sess, args.model_root)), flush=True)

def logistic_regression():

    args = init()
    w, h = 512, 660
    # Input
    X = tf.placeholder(shape=[args.batch_size, w, h, 1], dtype=tf.float32, name="x_input")
    target = tf.placeholder(shape=[args.batch_size, 1], dtype=tf.float32, name="y")
    print(X, flush=True)
    print(target, flush=True)

    flat = tf.reshape(X, (-1, w*h))
    """
    fc1 = tf.add(tf.matmul(flat, 
                weight_var([int(flat.get_shape()[1]), 1024])),
                    bias_var([1024]), name='fc1')
    print(fc1, flush=True)
    fc2 = tf.add(tf.matmul(fc1,
                weight_var([int(fc1.get_shape()[1]), 128])),
                    bias_var([128]), name='fc2')
    print(fc2, flush=True)
    """
    output = tf.add(tf.matmul(flat,
                weight_var([int(flat.get_shape()[1]), 1])),
                    bias_var([1]), name='output')
    
    pred = tf.sigmoid(output, name='pred')
    print(pred, flush=True)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred, target), tf.float32))
    #loss = tf.reduce_sum(tf.pow(tf.reduce_mean(pred)-target, 2))/(2*args.batch_size)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output,
                labels=target), name='loss')
    print(loss, flush=True)

    global_step = tf.Variable(args.init_epoch, trainable=False)
    global_epoch = args.init_epoch

    lr = tf.train.exponential_decay(
         learning_rate=args.lr,
         global_step=global_step,
         decay_steps=1e+5,
         decay_rate=args.decay_rate,
         staircase=True)
    opt = tf.train.AdamOptimizer(lr).minimize(
        loss,
        global_step=global_step)

    """
    for v in tf.trainable_variables():
        print(v.name, v.get_shape())
        if len(v.get_shape()) == 4:
          tf.summary.image(v.name, tf.reshape(v, (int(v.get_shape()[3])*int(v.get_shape()[2]), int(v.get_shape()[0]), int(v.get_shape()[1]), 1)))
        tf.summary.histogram(v.name, v)
    """
    grad = tf.gradients(loss, X)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.scalar('lr', lr)
    tf.summary.histogram('grad', grad)
    tf.summary.histogram('pred', pred)
    tf.summary.histogram('target', target)
    tf.summary.image('X', tf.reshape(X, (args.batch_size, int(X.get_shape()[1]), int(X.get_shape()[2]), 1)))

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
  
#          x[np.where(x < 1000)] = 0
#          x = x/1e+3

#          x = l2_norm(x)
          x = x.reshape(args.batch_size, w, h, 1)
          cnt = 10 if y==[1] else 1

          for _ in range(cnt):
            _, loss_val, acc_val, pred_val, summary = sess.run([
                opt, loss, acc, pred, merged], feed_dict={
                   X: x,
                   target: y.reshape(args.batch_size, 1)
                })
            train_writer.add_summary(summary, sess.run(global_step))

            print("[{}/{}]loss: {} acc: {} target: {} pred: {} output: {}".format(
              global_epoch,
              sess.run(global_step),
              loss_val,
              acc_val,
              y,
              pred_val,
              -0.,
              ), flush=True)
        print('model saved @ {}'.format(saver.save(sess, args.model_root)), flush=True)
        

def start():
  conv_dense()
  #logistic_regression()

if __name__ == '__main__':
  start()
