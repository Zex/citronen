# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
sys.path.insert(0, "/home/zex/anaconda3/lib/python3.6/site-packages")
sys.path.insert(0, os.getcwd()) 
import numpy as np
import pandas as pd
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BCELoss
from torch import optim
from torch import from_numpy
from torch.autograd import Variable
from iceberg.iceberg import Iceberg, Mode


class Torch(Iceberg, Module):
    def __init__(self, args):
        super(Torch, self).__init__(args)
        self.total_class = 2

        self.model = Sequential(
                Conv2d(11250, 1024,
                    kernel_size=5,
                    stride=2,
                    bias=True),
                MaxPool2d(kernel_size=3),
                ReLU(False),
                Conv2d(1024, 512,
                    kernel_size=5,
                    stride=2,
                    bias=True),
                MaxPool2d(kernel_size=3),
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
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        for e in range(1, self.epochs+1):
            self.foreach_epoch(e)
            if e % 100:
                torch.save({
                    'epoch': e,
                    'model': self.model,
                    'opt': optimizer,
                    }, 'iceberg-torch-{}'.format(e))

    def foreach_epoch(self, e):
        X, y = self.preprocess()
        X = np.array(X)
        """
        # 1604, 11250
        """
        X = X.reshape(1604, 11250, 1, 1)
        output = self(Variable(from_numpy(X)))
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
