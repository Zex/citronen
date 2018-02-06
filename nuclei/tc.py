# Nuclei TORCH:w
# Author: Zex Li <top_zlynch@yahoo.com>
from torch import from_numpy
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d
from torch.nn import MaxUnpool2d
from torch.nn import Upsample
from nuclei.provider import *


class TC(Module):

    def __init__(self):
        super(TC, self).__init__()
        self._build_model()


    def _build_model(self):
        self.model = Sequential(
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            MaxPool2d(kernel_size=(2, 2), stride=1),
            ReLU(),
            #MaxUnpool2d(kernel_size=(2, 2), stride=1),
            Upsample(scale_factor=2),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            #MaxUnpool2d(kernel_size=(2, 2), stride=1),
            Upsample(scale_factor=2),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            #MaxUnpool2d(kernel_size=(2, 2), stride=1),
            Upsample(scale_factor=2),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
            Conv2d(4, 4,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
                bias=True),
        )

    def forward(self, x):
        x = self.model(x)


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
#        self.device = "/cpu:0"

    def _build_model(self):
        self.model = TC()

    def train(self):
        self._build_model()

        for e in range(self.epochs):
            self.foreach_epoch(e)

    def foreach_epoch(self, e):
        for X, y, total_nuclei in self.prov.gen_data():
            X = X.reshape(X.shape[0], X.shape[-1], *X.shape[1:-1]).astype(np.float32)
            X = Variable(from_numpy(X))
            output = self.model(X)
            #print(output)
            plt.imshow(output)
            plt.show()
            break
            
    def inner_test(self, step):
        pass


def start():
    tc = Runner(init())
    tc.train()


if __name__ == '__main__':
    start()
