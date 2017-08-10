from common import init, plot_img, init_axs, data_generator, reinit_plot
import tensorflow as tf
import numpy as np
from os import mkdir
from os.path import isdir


def xavier_init(size):
    stddev = 1./np.sqrt(size[0]/2)
    return tf.Variable(tf.random_normal(shape=size, stddev=stddev))

def rand_sample(size):
    return np.random.uniform(-1., 1., size=size)

eps = 1e-8
w, h = 512, 660
batch, size = 16, w*h
x_dim = w
gen_sample_nr = 16


dw1 = xavier_init([size, w])
db1 = tf.Variable(tf.zeros(shape=[w]))
dw2 = xavier_init([w, 1])
db2 = tf.Variable(tf.zeros(shape=[1]))

gw1 = xavier_init([size, w])
gb1 = tf.Variable(tf.zeros(shape=[w]))
gw2 = xavier_init([w, size])
gb2 = tf.Variable(tf.zeros(shape=[size]))

gen_params = [gw1, gb1, gw2, gb2]
dis_params = [dw1, db1, dw2, db2]

def gen(x):
    x = tf.nn.relu(tf.matmul(x, gw1) + gb1)
    return tf.nn.sigmoid(tf.matmul(x, gw2) + gb2)

def dis(x):
    x = tf.nn.relu(tf.matmul(x, dw1) + db1)
    x = tf.matmul(x, dw2) + db2
    y = tf.nn.sigmoid(x)
    return x, y

gen_data = tf.placeholder(tf.float32, shape=[None, size])
real_x = tf.placeholder(tf.float32, shape=[None, size])
real_logist, real_y = dis(real_x)
fake_x = gen(gen_data)
fake_logist, fake_y = dis(fake_x)

real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(real_logist), labels=tf.ones_like(real_logist)))
fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(fake_logist), labels=tf.zeros_like(fake_logist)))
dis_loss = real_loss + fake_loss
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(fake_logist), labels=tf.ones_like(fake_logist)))

opt_dis = tf.train.AdamOptimizer().minimize(dis_loss, var_list=dis_params)
opt_gen = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_params)


def train():
    args = init()
    global_epoch = args.init_epoch
    gen_path = args.outpath

    if not isdir(gen_path):
        mkdir(gen_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(args.init_epoch, args.epochs):
            global_epoch += 1
            for i, (data, y) in enumerate(data_generator(args. data_root, args.label_path)):
                if y.size == 0:
                    continue

                data[np.where(data < 10000)] = 0.
                data = sess.run(tf.nn.l2_normalize(data.astype(np.float32), 0, epsilon=eps))
                np.save('{}/{}'.format(gen_path, 0), data)

                rand = sess.run(tf.nn.l2_normalize(rand_sample([batch, size]), 0, epsilon=eps))
                _, _dis_loss = sess.run([opt_dis, dis_loss], feed_dict={
                    real_x: data,
                    gen_data: rand_sample([batch, size])
                })
                rand = sess.run(tf.nn.l2_normalize(rand_sample([batch, size]), 0, epsilon=eps))
                _, _gen_loss = sess.run([opt_gen, gen_loss], feed_dict={
                    gen_data: rand_sample([batch, size])
                })
                if _dis_loss is None or _gen_loss is None:
                    print('[{}/{}] dis_loss:{} gen_loss:{}'.format(
                        global_epoch, i+1, _dis_loss, _gen_loss), flush=True)
                    return

                print('[{}/{}] dis_loss:{:.4} gen_loss:{:.4}'.format(
                    global_epoch, i+1, _dis_loss, _gen_loss), flush=True)
                
                sample = sess.run(fake_x, feed_dict={
                    gen_data: rand_sample([batch, size])
                })
                ind = global_epoch%gen_sample_nr
                ind and np.save('{}/{}'.format(gen_path, ind), sample) or None
            

if __name__ == '__main__':
  train()



