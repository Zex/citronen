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
from torch import from_numpy, save
from torch.autograd import Variable
from iceberg.iceberg import Iceberg, Mode


class Torch(Iceberg, Module):
    def __init__(self, args):
        super(Torch, self).__init__(args)
        self.total_class = 2

        self.features = Sequential(
                Conv2d(1, 1024,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=True),
                MaxPool2d(kernel_size=2),
                ReLU(False),
                Conv2d(1024, 512,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    bias=True),
                MaxPool2d(kernel_size=3),
                ReLU(False),
                )
        self.classifier = Sequential(
                Linear(512*3*3, self.total_class),
                #Dropout(0.2, False),
                )

    def forward(self, x):
        x = self.features(x)
        print("feature",  x.data.numpy().shape)
        # 1604, 512, 3, 3
        x = x.view(x.size(0), -1)
        print("feature",  x.data.numpy().shape)
        x = self.classifier(x)
        print('feature', x.data.numpy().shape)
        return x

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"
        self.loss_fn = BCELoss().cpu()
        self.optimizer = optim.Adam(self.parameters(), self.lr)

        for e in range(1, self.epochs+1):
            self.foreach_epoch(e)
            if e % 100:
                save({
                    'epoch': e,
                    'model': self.model,
                    'opt': self.optimizer,
                    }, 'iceberg-torch-{}'.format(e))

    def foreach_epoch(self, e):
        X, y = self.preprocess()
        X = np.array(X)
        """
        # 1604, 5625
        """
        def foreach_yi(yi):
            i = np.zeros(2)
            i[yi] = 1
            yi = i
            return yi

        X = X.reshape(1604, 1, 75, 75).astype(np.float32)
        X = Variable(from_numpy(X))
        y = np.array(list(map(lambda yi: foreach_yi(yi), y))).astype(np.float32)
        y = Variable(from_numpy(y))
        output = self(X)
        output = F.sigmoid(output)
        print("++ [epoch-{}] output:{}".format(e, output.data.numpy().tolist()) )
        pred = F.binary_cross_entropy_with_logits(output, y)
        loss = self.loss_fn(output, y)
        print("++ [epoch-{}] loss:{}".format(e, loss.data.numpy().tolist())) 
        loss.backward()
        self.optimizer.step()

    def test(self):
        pass

    def eval(self):
        pass


if __name__ == '__main__':
    Torch.start()
