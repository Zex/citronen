# Driver Insurance Prediction
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import ujson
import string
import numpy as np
import argparse
import pandas as pd


class Mode(object):
    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'


class DriverInsurance(object):

    def __init__(self, args):
        super(DriverInsurance, self).__init__()
        self.lr = 1e-2
        self.batch_size = args.batch_size
        self.epochs = 1000
        self.model = None
        self.model_dir = args.model_dir
        self.eval_result_path = os.path.join(self.model_dir, 'logs', 'eval.json')
        self.has_model = args.load_model
        self.error_stop_cnt = 0
        self.last_epoch = 0

        base = os.path.dirname(self.eval_result_path)
        if not os.path.isdir(base):
            os.makedirs(base)
    
    def load_data(self, path):
        if not os.path.isfile(path):
            return None
        data = pd.read_csv(path)
        return data
   
    def analyze(self):
        self.path = "data/driver_insurance/train.csv"
        self.mode = Mode.TRAIN
        data = self.preprocess()

        self.row = 1; self.col = 1
        self.axes = []
        self.fig = plt.figure(figsize=(6,6), facecolor='gray', edgecolor='black')

        for i, feat in enumerate(data.keys()):
            if feat in ('id', 'target'):
                continue
            print(data[feat], feat)
            self.plot_feature(data, feat)

    def get_ax(self, i):
        cur = i%(self.row*self.col)+1

        if len(self.axes) > cur:
            ax = self.axes[cur]
            ax.clear()
        else:
            ax = self.fig.add_subplot(self.row, self.col, i%(self.row*self.col)+1)
            self.axes.append(ax)
        return ax

    def plot_feature(self, i, data, feat):
        ax = self.get_ax(i)
        ax.plot(range(len(data[feat])), data[feat], 'bo', markersize=3.0)
        ax.annotate('F {}'.format(feat), xy=(1,6))
        self.fig.canvas.draw()

    def preprocess(self):
        data = self.load_data(self.path)
        X = list(map(lambda f: data[f].values, data.keys()))
        X = np.array(X)

        if self.mode in (Mode.TRAIN, Mode.EVAL):
            y = data['target'].values.reshape(len(data['target']), 1).astype(np.float)
            """
            X: (59, 595212)
            y: (595212, 1)
            """
            return X, y
        return iid.values, X

    def load_model(self):
        pass

    @classmethod
    def init(cls):
        parser = argparse.ArgumentParser(description="Driver Insurance Prediction")
        parser.add_argument('--train', action='store_true', default=False, help='train a model')
        parser.add_argument('--test', action='store_true', default=False, help='test a model')
        parser.add_argument('--eval', action='store_true', default=False, help='eval a model')
        parser.add_argument('--anal', action='store_true', default=False, help='analyse data')
        parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
        parser.add_argument('--load_model', action='store_true', default=False, help='load exist model')
        parser.add_argument('--model_dir', type=str, default='models/driver_insurance', help='model directory')
        return parser.parse_args()

    @classmethod
    def start(cls):
        args = cls.init()
        drv_ins = cls(args)
    
        if args.train:
            drv_ins.train()
        if args.test:
            drv_ins.test()
        if args.eval:
            drv_ins.eval()
        if args.anal:
            drv_ins.analyze()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    plt.ion()
    DriverInsurance.start()
