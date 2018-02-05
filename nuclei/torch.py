# Nuclei TORCH:w
# Author: Zex Li <top_zlynch@yahoo.com>
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import MaxUnpool2d
from nuclei.provider import *


class TC(Module):

    def __init__(self):
        super(TC, self).__init__()


    def _build_model(self):
        self.model = Squential(
            Conv2d(4, 5,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True)
            MaxPool2d(kernel_size=(3, 3),
            ReLU(),
            Conv2d(5, 5,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True)
            MaxPool2d(kernel_size=(3, 3),
            ReLU(),
            Conv2d(5, 5,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True)
            MaxPool2d(kernel_size=(3, 3),
            ReLU(),
            MaxUpsample2d(kernel_size=3, 3),
            MaxUpsample2d(kernel_size=3, 3),
            MaxUpsample2d(kernel_size=3, 3),
            )

    def forward(self, x):
        x = self.model(x)
        print('output', x.data.numpy().shape)


class Runner(object):

    def __init__(self, args):
        super(TF, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.init_step = args.init_step
        self.dropout_rate = args.dropout_rate
        self.summ_intv = args.summ_intv
        self.model_dir = args.model_dir
        self.log_path = os.path.join(self.model_dir, 'cnn')

        self.prov = Provider()
        self.height, self.width = self.prov.height, self.prov.width
        self.channel = self.prov.channel
        self.device = "/cpu:0"

    def _build_model(self):
        self.model = TC()

    def train(self):
        self._build_model()
        #self.estimator.train(input_fn=preprocess, steps=100)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)
            self.saver = tf.train.Saver(tf.global_variables())

            for e in range(self.epochs):
                self.foreach_epoch(sess, e)

    def foreach_epoch(self, sess, e):
        for X, y, total_nuclei in self.prov.gen_data():
            feed_dict = {
                self.input_x: X,
                self.input_y: y,
                self.dropout_keep: 1-self.dropout_rate,
                self.total_nuclei: total_nuclei,
                }

            _, loss, step, summ = sess.run(\
                [self.train_op, self.loss, self.global_step, self.summary],\
                feed_dict=feed_dict)

            if step % self.summ_intv == 0:
                print('++ [step-{}] loss:{}'.format(step, loss), flush=True)
                self.summary_writer.add_summary(summ, step)
                self.saver.save(sess, self.model_dir+'/cnn',
                            global_step=tf.train.global_step(sess, self.global_step))
                self.inner_test(sess, step)
        
                
    def inner_test(self, sess, step):
        pass


def start():
    tc = Runner(init())
    tc.train()


if __name__ == '__main__':
    start()
