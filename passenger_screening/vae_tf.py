from common import init, plot_img, init_axs, data_generator, reinit_plot
import tensorflow as tf
import numpy as np
from os import mkdir
from os.path import isdir


def xavier_init(x_dim, name):
    stddev = 1./np.sqrt(x_dim[0]/2)
    return tf.Variable(tf.random_normal(shape=x_dim, stddev=stddev), name=name)

def rand_sample(mu, sig):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(eps/2) * eps

eps = 1e-8
w, h = 512, 660
batch, x_dim = 1, w*h
gen_sample_nr = 100
z_dim = 256

qw = xavier_init([x_dim, batch], 'qw')
qb = tf.Variable(tf.zeros(shape=[batch]), name='qb')

qmu_w = xavier_init([batch, z_dim], 'mu_w')
qmu_b = tf.Variable(tf.zeros(shape=[z_dim]), name='mu_b')

qsigma_w = xavier_init([z_dim, batch], 'sigma_w')
qsigma_b = tf.Variable(tf.zeros(shape=[batch]), name='sigma_b')

pw1 = xavier_init([z_dim, batch], 'pw1')
pb1 = tf.Variable(tf.zeros(shape=[batch]), name='pb1')

pw2 = xavier_init([batch, x_dim], 'pw2')
pb2 = tf.Variable(tf.zeros(shape=[x_dim]), name='pb2')

for v in tf.trainable_variables():
    print(v)

def q(x):
    x = tf.nn.relu(tf.matmul(x, qw) + qb)
    mu = tf.matmul(x, qmu_w) + qmu_b
    sig = tf.matmul(mu, qsigma_w) + qsigma_b
    return mu, sig

def p(x):
    x = tf.nn.relu(tf.matmul(x, pw1) + pb1)
    x = tf.matmul(x, pw2) + pb2
    y = tf.nn.sigmoid(x)
    return x, y

real_x = tf.placeholder(tf.float32, shape=[batch, x_dim], name='real_x')
z = tf.placeholder(tf.float32, shape=[batch, z_dim], name='z')

mu, sig = q(real_x)
gen_data = rand_sample(mu, sig)
_, logits = p(gen_data)
fake_x, _ = p(z)

recon_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=real_x), 1)
kl_loss = 0.5 * tf.reduce_sum(tf.exp(sig) + mu**2 - 1. - sig, 1)
vae_loss = tf.reduce_mean(recon_loss + kl_loss)
print('logits', logits, flush=True)
print('fake_x', fake_x, flush=True)
print('recon_loss', recon_loss, flush=True)
print('kl_loss', kl_loss, flush=True)
print('vae_loss', vae_loss, flush=True)

opt = tf.train.AdamOptimizer().minimize(vae_loss)
tf.summary.histogram('sig', sig)
tf.summary.histogram('mu', mu)
#tf.summary.scalar('recon_loss', recon_loss)
#tf.summary.scalar('kl_loss', kl_loss)
tf.summary.scalar('vae_loss', vae_loss)
tf.summary.image('real_x', tf.reshape(real_x, [batch, w, h, 1]))
tf.summary.image('z', tf.reshape(z, [batch, 16, 16, 1]))
tf.summary.image('gen_data', tf.reshape(gen_data, [batch, 16, 16, 1]))
tf.summary.image('logits', tf.reshape(logits, [batch, w, h, 1]))
tf.summary.image('fake_x', tf.reshape(fake_x, [batch, w, h, 1]))
merged = tf.summary.merge_all()
saver = tf.train.Saver(tf.global_variables())

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
                if y.shape == 0:
                    continue

                data = data.reshape(batch, x_dim)

                _, _vae_loss, summary = sess.run([opt, vae_loss, merged], feed_dict={
                    real_x: data,
                    z: np.random.randn(batch, z_dim)
                })
                train_writer.add_summary(summary, global_steps)

                if _vae_loss is None:
                    print('[{}/{}] vae_loss:{}'.format(global_epoch, i+1, _vae_loss), flush=True)
                    return

                print('[{}/{}] vae_loss:{:.4}'.format(global_epoch, i+1, _vae_loss), flush=True)

                global_steps += 1
                if i//gen_sample_nr != 0:
                    continue

                sample = sess.run(fake_x, feed_dict={
                    z: np.random.randn(batch, z_dim)
                })

                ind = i%gen_sample_nr
                i//gen_sample_nr and np.save('{}/{}'.format(gen_path, ind), sample) or None
            print('model saved @ {}'.format(saver.save(sess, args.model_root)), flush=True)


if __name__ == '__main__':
  train()
