# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from iceberg.iceberg import Iceberg


class Torch(Iceberg, Module):

    def __init__(self, args):
        super(Torch, self).__init__(args)
        self.total_class = 2
        self.classifier = Sequential(
                Conv2d(75*75, 1024,
                    kernel_size=5,
                    stride=2,
                    bias=True),
                MaxPool1d(kernel_size=3),
                ReLU(False),
                Conv2d(1024, 512,
                    kernel_size=5,
                    stride=2,
                    bias=True),
                MaxPool1d(kernel_size=3),
                ReLU(False),
                Linear(512, self.total_class),
                Dropout(0.2, False),
                ReLU(False),
                )

    def forward(self, x):
        x = self.model(x)
        return x

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"
        self.loss_fn = BCELoss().cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        for e in self.total_epochs:
            self.foreach_epoch(e)
            if e % 100:
                torch.save({
                    'epoch': e,
                    'model': self.model,
                    'opt': optimizer,
                    }, 'iceberg-torch-{}'.format(e))

    def foreach_epoch(self, e):
        X, y = self.preprocess()
        output = self()
        pred = F.binary_cross_entropy(output, y)
        print("++ [epoch-{}] output:{} lbl:{}".format(e, output, y)) 
        loss = self.loss_fn(output, y)
        print("++ [epoch-{}] loss:{} lbl:{}".format(e, loss, y)) 
        loss.backward()
        optimizer.step()

    def test(self):
        pass

    def eval(self):
        pass


if __name__ == '__main__':
    Torch.start()
