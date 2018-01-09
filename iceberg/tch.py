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
from torch.nn import BatchNorm2d
from torch.nn import BCELoss
from torch.nn import Sigmoid
from torch import optim
from torch import from_numpy, save
from torch.autograd import Variable
from sklearn.metrics import log_loss, roc_auc_score
from iceberg.iceberg import Iceberg, Mode


class Torch(Iceberg, Module):
    def __init__(self, args):
        super(Torch, self).__init__(args)
        self.total_class = 2
        self.batch_size = 32

        self.features = Sequential(
                Conv2d(1, 1024,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=True),
                BatchNorm2d(1024),
                MaxPool2d(kernel_size=2),
                Sigmoid(),
                Conv2d(1024, 512,
                    kernel_size=5,
                    padding=2,
                    stride=2,
                    bias=True),
                BatchNorm2d(512),
                MaxPool2d(kernel_size=3),
                Sigmoid(),
                )
        self.classifier = Sequential(
                Linear(512*3*3, 1), #self.total_class),
                Sigmoid(),
                #Dropout(0.2, False),
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"
        self.loss_fn = BCELoss()
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
        #y = np.where(y == 0,[1,0],[0,1])
        cur = 0
        while cur < len(X):
            self.foreach_batch(e, X[cur:cur+self.batch_size], y[cur:cur+self.batch_size])
            cur += self.batch_size

    def foreach_batch(self, e, X, y):
        X = X.reshape(self.batch_size, 1, 75, 75).astype(np.float32)
        X = Variable(from_numpy(X), requires_grad=True)

        y = y.astype(np.float32)
        y = Variable(from_numpy(y))

        output = self(X)
        loss = self.loss_fn(output, y)
        print("++ [epoch-{}] y:{}".format(e, y.data.numpy().tolist()))
        print("++ [epoch-{}] output:{}".format(e, output.data.numpy().tolist()))
        print("++ [epoch-{}] loss:{}".format(e, loss.data.numpy().tolist())) 

        #pred = F.binary_cross_entropy_with_logits(output, y)
        pred = output.float()
        y_squeeze = y.data.numpy().squeeze()
        pred_squeeze = pred.data.numpy().squeeze()
        """
        logloss = log_loss(y_squeeze, pred_squeeze, eps=1e-12)
        score = roc_auc_score(y_squeeze, pred_squeeze)
        print("++ [epoch-{}] pred:{}".format(e, pred_squeeze.tolist()))
        print("++ [epoch-{}] logloss:{} score:{}".format(e, logloss, score))
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self):
        pass

    def eval(self):
        pass


if __name__ == '__main__':
    Torch.start()
