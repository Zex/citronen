from common import init, plot_img, init_axs, data_generator, reinit_plot
import tensorflow as tf
import numpy as np
from os import mkdir
from os.path import isdir


def xavier_init(size, name):
    stddev = 1./np.sqrt(size[0]/2)
    return tf.Variable(tf.random_normal(shape=size, stddev=stddev), name=name)

def rand_sample(size):
    return np.random.uniform(-1., 1., size=size).astype(np.float32)

eps = 1e-8
w, h = 512, 660
batch, size = 1, w*h
x_dim = w
gen_sample_nr = 100


dw1 = xavier_init([size, w], 'dw1')
db1 = tf.Variable(tf.zeros(shape=[w]), name='db1')
dw2 = xavier_init([w, 1], 'dw2')
db2 = tf.Variable(tf.zeros(shape=[1]), name='db2')

gw1 = xavier_init([size, w], 'gw1')
gb1 = tf.Variable(tf.zeros(shape=[w]), name='gb1')
gw2 = xavier_init([w, size], 'gw2')
gb2 = tf.Variable(tf.zeros(shape=[size]), name='gb2')

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

gen_data = tf.placeholder(tf.float32, shape=[batch, size], name='gen_data')
real_x = tf.placeholder(tf.float32, shape=[batch, size], name='real_x')
real_logist, real_y = dis(real_x)
fake_x = gen(gen_data)
fake_logist, fake_y = dis(fake_x)

real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logist, labels=tf.ones_like(real_logist)))
fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logist, labels=tf.zeros_like(fake_logist)))
dis_loss = real_loss + fake_loss
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logist, labels=tf.ones_like(fake_logist)))

opt_dis = tf.train.AdamOptimizer().minimize(dis_loss, var_list=dis_params)
opt_gen = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_params)

tf.summary.scalar('real_loss', real_loss)
tf.summary.scalar('fake_loss', fake_loss)
tf.summary.scalar('dis_loss', dis_loss)
tf.summary.scalar('gen_loss', gen_loss)
tf.summary.image('real_x', tf.reshape(real_x, [batch, w, h, 1]))
tf.summary.image('gen_data', tf.reshape(gen_data, [batch, w, h, 1]))
tf.summary.image('fake_x', tf.reshape(fake_x, [batch, w, h, 1]))
merged = tf.summary.merge_all()

for v in tf.trainable_variables():
    print(v)
    if len(v.get_shape()) == 2:
        tf.summary.image(v.name, tf.reshape(v, [1,
            int(v.get_shape()[0]), int(v.get_shape()[1]), 1]))
    tf.summary.histogram(v.name, v)

def train():
    args = init()
    global_epoch = args.init_epoch
    global_steps = 0 
    gen_path = args.outpath

    if not isdir(gen_path):
        mkdir(gen_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(args.model_root, sess.graph)

        for e in range(args.init_epoch, args.epochs):
            global_epoch += 1
            for i, (data, y) in enumerate(data_generator(args. data_root, args.label_path)):
                if y.size == 0:
                    continue

                #data[np.where(data < 255)] = 0.
                #data = data.astype(np.float32)
                data = data.reshape(batch, size)

                _, _dis_loss, summary = sess.run([opt_dis, dis_loss, merged], feed_dict={
                    real_x: data,
                    gen_data: rand_sample([batch, size])
                })
                train_writer.add_summary(summary, global_steps)

                _, _gen_loss, summary = sess.run([opt_gen, gen_loss, merged], feed_dict={
                    real_x: data,
                    gen_data: rand_sample([batch, size])
                })
                train_writer.add_summary(summary, global_steps)

                if _dis_loss is None or _gen_loss is None:
                    print('[{}/{}] dis_loss:{} gen_loss:{}'.format(
                        global_epoch, i+1, _dis_loss, _gen_loss), flush=True)
                    return

                print('[{}/{}] dis_loss:{:.4} gen_loss:{:.4}'.format(
                    global_epoch, i+1, _dis_loss, _gen_loss), flush=True)

                sample = sess.run(fake_x, feed_dict={
                    gen_data: rand_sample([batch, size])
                })

                ind = i%gen_sample_nr
                i//gen_sample_nr and np.save('{}/{}'.format(gen_path, ind), sample) or None
                global_steps += 1


if __name__ == '__main__':
  train()
