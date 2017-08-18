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
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv_dense():
    args = init()
    w, h = 512, 660
    dim = w*h

    # Input
    X = tf.placeholder(shape=[args.batch_size, w, h, 1], dtype=tf.float32, name="x_input")
    target = tf.placeholder(shape=[args.batch_size, 1], dtype=tf.float32, name="y")
    print(X, flush=True)
    print(target, flush=True)

    # Conv1
    conv1 = tf.nn.relu(tf.nn.conv2d(X, weight_var([4, 4, 1, 512]), [1, 2, 2, 1], 
                padding='SAME', name='conv1') + bias_var([512]))
    norm1 = tf.nn.lrn(conv1, depth_radius=5, bias=1.0, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    print(pool1, flush=True)

    # Conv2
    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weight_var([4, 4, 512, 128]), [1, 2, 2, 1],
                padding='SAME', name='conv2') + bias_var([128]))
    norm2 = tf.nn.lrn(conv2, depth_radius=5, bias=1.0, name='norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print(pool2, flush=True)

    # Conv3
    conv3 = tf.nn.relu(tf.nn.conv2d(pool2, weight_var([3, 3, 128, 32]), [1, 2, 2, 1],
                padding='SAME', name='conv3') + bias_var([32]))
    norm3 = tf.nn.lrn(conv3, depth_radius=5, bias=1.0, name='norm3')
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    print(pool3, flush=True)

    # Conv4
    conv4 = tf.nn.relu(tf.nn.conv2d(pool3, weight_var([3, 3, 32, 8]), [1, 2, 2, 1],
                padding='SAME', name='conv4') + bias_var([8]))
    norm4 = tf.nn.lrn(conv4, depth_radius=5, bias=1.0, name='norm4')
    pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    print(pool4, flush=True)

    #flat = tf.reshape(pool4, [-1, 2*3*512])
    flat = tf.reshape(pool4, [-1, 2*3*8])

    dense1 = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.elu, use_bias=True, name='dense1')
    print(dense1, flush=True) 

    dense2 = tf.layers.dense(inputs=dense1, units=2, activation=tf.nn.elu, use_bias=True, name='dense2')
    print(dense2, flush=True) 

    """
    pred = tf.cast(tf.equal(dense2, target), tf.float32)#tf.argmax(dense2, 0), tf.argmax(target, 0))
    acc = tf.reduce_mean(pred)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
        labels=target,
        logits=dense2,
        #pos_weight=.1,
        name='loss'
        ))
    """
    flat = tf.reshape(X, (-1, w*h))
    fc1 = tf.add(tf.matmul(flat, 
                weight_var([int(flat.get_shape()[1]), 1024])), #int(flat.get_shape()[0])])),
                    bias_var([1024]), name='fc1')
    print(fc1, flush=True)
    fc2 = tf.add(tf.matmul(fc1, 
                weight_var([int(fc1.get_shape()[1]), 256])), #int(fc1.get_shape()[0])])),
                    bias_var([256]), name='fc2')
    print(fc2, flush=True)
    pred = tf.add(tf.matmul(fc2,
                weight_var([int(fc2.get_shape()[1]), 1])),
                    bias_var([1]), name='pred')
    print(pred, flush=True)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred, target), tf.float32))
    loss = tf.reduce_sum(tf.pow(tf.reduce_mean(pred)-target, 2))/(2*args.batch_size)
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

    for v in tf.trainable_variables():
        print(v.name, v.get_shape())
        if len(v.get_shape()) == 4:
          tf.summary.image(v.name, tf.reshape(v, (int(v.get_shape()[3])*int(v.get_shape()[2]), int(v.get_shape()[0]), int(v.get_shape()[1]), 1)))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    tf.summary.scalar('lr', lr)
    tf.summary.image('X', tf.reshape(X, (args.batch_size, int(X.get_shape()[2]), int(X.get_shape()[1]), 1)))
    #tf.summary.image('dense2', tf.reshape(dense2, (args.batch_size, int(dense2.get_shape()[2]), int(dense2.get_shape()[1]), 1)))
    #tf.summary.image('pool3', tf.reshape(pool3, (args.batch_size*32, int(pool3.get_shape()[2]), int(pool3.get_shape()[1]), 1)))
    #tf.summary.image('pool4', tf.reshape(pool4, (args.batch_size*8, int(pool4.get_shape()[2]), int(pool4.get_shape()[1]), 1)))

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
  
          x[np.where(x < 255)] = 0
          #x[np.where(x >= 255)] = x[np.where(x >= 255)]/10000
          x = x/1e+3

#          x = l2_norm(x)
          x = x.reshape(args.batch_size, w, h, 1)#.astype(np.float32)
 
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
              np.mean(y),
              np.mean(pred_val),
              -0.,
              ), flush=True)
        print('model saved @ {}'.format(saver.save(sess, args.model_path)), flush=True)
        

def start():
  conv_dense()

if __name__ == '__main__':
  start()
