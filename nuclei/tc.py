# Nuclei TORCH:w
# Author: Zex Li <top_zlynch@yahoo.com>
from datetime import datetime
from torch import from_numpy, cuda
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import MaxUnpool2d
from torch.nn import Upsample
from torch.optim import Adam
from torch.nn import MSELoss
import torch
from nuclei.provider import *


class TC(Module):

    def __init__(self):
        super(TC, self).__init__()
        self._build_model()


    def _build_model(self):
        self.model = Sequential(
            Conv2d(4, 64,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(64, 64,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=2),
            ReLU(),
            Conv2d(64, 128,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(128, 128,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=2),
            ReLU(),
            Conv2d(128, 256,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(256, 256,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),
            Conv2d(256, 512,
                kernel_size=(3, 3),
                stride=2,
                padding=0,
                bias=True),
            Conv2d(512, 512,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),

            Conv2d(512, 1024,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(1024, 1024,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            ReLU(),
            #MaxUnpool2d(kernel_size=(2, 2), stride=2),
            Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(1024, 512,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(512, 512,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            ReLU(),
            #MaxUnpool2d(kernel_size=(2, 2), stride=2),
            Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(512, 256,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(256, 256,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            ReLU(),
            #MaxUnpool2d(kernel_size=(2, 2), stride=2),
            Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(256, 128,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(128, 128,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            ReLU(),
            Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(128, 64,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(64, 64,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(64, 1,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            ReLU(),
        )

    def forward(self, x):
        x = self.model.cuda(x) if cuda.is_available() else self.model(x)
        return x#.data.numpy()


class Runner(object):

    def __init__(self, args):
        super(Runner, self).__init__()
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
        self.global_step = 0
        self.data_path = "data/nuclei/predict"
        self.loss_fn = MSELoss()
#        self.device = "/cpu:0"

    def _build_model(self):
        self.model = TC()
        self.optimizer = Adam(self.model.parameters(), self.lr)
        print('++ [info] cuda available: {}'.format(cuda.is_available()))
        print('++ [info] parameters')
        list(map(lambda p: print('param', p.shape), self.model.parameters()))
        #list(map(lambda p: print('mod', p), self.model.modules()))

    def train(self):
        self._build_model()

        for e in range(self.epochs):
            self.foreach_epoch(e)

    def loss_fn(self, pred, target):
        alpha = 1e-12
        loss = np.sum(list(map(\
                lambda z:1.0-z[0]*z[1]/(alpha+z[0]), zip(target, pred))))/len(target)
        return loss

    def foreach_epoch(self, e):
        for X, y, total_nuclei in self.prov.gen_data():
            self.global_step += 1
            X = X.reshape(X.shape[0], X.shape[-1], *X.shape[1:-1]).astype(np.float32)
            X = Variable(from_numpy(X))
            y = y.reshape(y.shape[0], y.shape[-1], *y.shape[1:-1]).astype(np.float32)
            y = Variable(from_numpy(y))
            output = self.model(X)
            print("++ [output] {}".format(output))
            if self.global_step % 100 == 0:
                plt.imshow(np.squeeze(output))
                plt.imsave("{}/ouput_{}-{}.png".format(
                    self.data_path,
                    self.globa_step,
                    datetime.now().strftime("%y%m%d%H%M")))
            loss = self.loss_fn(output, y)
            print('++ [step/{}/{}] loss:{:.4f}'.format(\
                   self.global_step, \
                   datetime.now().strftime("%y%m%d%H%M%S"), \
                   np.squeeze(loss.data.numpy())))
            loss.backward()
            self.optimizer.step()

    def save(self):
      torch.save({
          'epoch': global_epoch,
          'model': self.model,
          'optimizer': self.optimizer,
          }, '{}/{}-{}'.format(self.model_dir, "torch", self.global_epoch))
            
    def inner_test(self, step):
        pass


def start():
    #args = init()
    tc = Runner(Config())
    tc.train()


if __name__ == '__main__':
    start()
