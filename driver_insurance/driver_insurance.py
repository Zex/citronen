# Driver Insurance Prediction
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import ujson
import string
import numpy as np
from datetime import datetime
import argparse
import pandas as pd
from sklearn import decomposition as decom
from sklearn import svm


class Mode(object):
    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'

header = [
    "ps_ind_01",
    "ps_ind_02_cat",
    "ps_ind_03",
    "ps_ind_04_cat",
    "ps_ind_05_cat",
    "ps_ind_06_bin",
    "ps_ind_07_bin",
    "ps_ind_08_bin",
    "ps_ind_09_bin",
    "ps_ind_10_bin",
    "ps_ind_11_bin",
    "ps_ind_12_bin",
    "ps_ind_13_bin",
    "ps_ind_14",
    "ps_ind_15",
    "ps_ind_16_bin",
    "ps_ind_17_bin",
    "ps_ind_18_bin",
    "ps_reg_01",
    "ps_reg_02",
    "ps_reg_03",
    "ps_car_01_cat",
    "ps_car_02_cat",
    "ps_car_03_cat",
    "ps_car_04_cat",
    "ps_car_05_cat",
    "ps_car_06_cat",
    "ps_car_07_cat",
    "ps_car_08_cat",
    "ps_car_09_cat",
    "ps_car_10_cat",
    "ps_car_11_cat",
    "ps_car_11",
    "ps_car_12",
    "ps_car_13",
    "ps_car_14",
    "ps_car_15",
    "ps_calc_01",
    "ps_calc_02",
    "ps_calc_03",
    "ps_calc_04",
    "ps_calc_05",
    "ps_calc_06",
    "ps_calc_07",
    "ps_calc_08",
    "ps_calc_09",
    "ps_calc_10",
    "ps_calc_11",
    "ps_calc_12",
    "ps_calc_13",
    "ps_calc_14",
    "ps_calc_15_bin",
    "ps_calc_16_bin",
    "ps_calc_17_bin",
    "ps_calc_18_bin",
    "ps_calc_19_bin",
    "ps_calc_20_bin",
]

top20 = {
     "f42": 2204,
     "f54": 2212,
     "f8": 2223,
     "f32": 2240,
     "f29": 2261,
     "f41": 2271,
     "f18": 2385,
     "f25": 2531,
     "f23": 2557,
     "f40": 2585,
     "f55": 2591,
     "f45": 2793,
     "f53": 2837,
     "f34": 2898,
     "f1": 3024,
     "f36": 3085,
     "f17": 3159,
     "f5": 3198,
     "f7": 3223,
     "f38": 4448,
}

class DriverInsurance(object):

    def __init__(self, args):
        super(DriverInsurance, self).__init__()
        self.lr = 1e-3
        self.batch_size = args.batch_size
        self.epochs = 1000
        self.model = None
        self.model_dir = args.model_dir
        self.eval_result_path = os.path.join(self.model_dir, 'logs', 'eval.json')
        self.result_path = "data/driver_insurance/pred_{}.csv".format(\
                datetime.now().strftime("%y%m%d%H%M"))
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
        data = self.load_data(self.path)
        X, y = self.preprocess()

        ana = svm.LinearSVC(penalty='l2', max_iter=1000, loss='squared_hinge', multi_class='ovr')
        ana.fit(X, y)
        pred = ana.predict(X)

        for iid, p in zip(data['id'], pred):
            print(iid, p)
        """
        ana = decom.PCA(n_components=20)
        comp = ana.fit_transform(X, y)
        print(comp)
        print(comp.shape)
        """
        self.row = 1
        self.col = 1
        self.axes = []

        #self.fig = plt.figure(figsize=(6, 6), facecolor='darkgray', edgecolor='black')
        #ax = self.get_ax(1)

        #ax.set_facecolor('black')
        #ax.plot(comp[0:10,0],comp[0:10,1], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
        #ax.plot(comp[10:20,0], comp[10:20,1], 'o', markersize=1, color='red', alpha=0.5, label='class2')
        #self.fig.canvas.draw()

        for i, feat in enumerate(data.keys()):
            if feat in ('id', 'target'):
                continue
            if feat not in KEYS:
                continue
            print(data[feat], feat)
            self.plot_feature(i-2, data, feat)

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
        ax.plot(range(len(data[feat])), data[feat], 'bo', markersize=1.0)
        ax.annotate('{}'.format(feat), xy=(1, 3))
        self.fig.canvas.draw()

    def preprocess(self):
        data = self.load_data(self.path)
        #X = list(map(lambda f: data[f].values, data.keys()))
        #X = [data[header[int(f[1:])]].values for f in top20]
        X = [data[f].values for f in data.keys() if f not in ('id', 'target')]
        X = np.array(X)
        X = X.reshape(X.shape[1], X.shape[0])

        if self.mode in (Mode.TRAIN, Mode.EVAL):
            y = data['target'].values.reshape(len(data['target']), 1).astype(np.float)
            """
            X: (59, 595212)
            y: (595212, 1)
            """
#            comp = ana.fit_transform(X, y)
#            print(comp)
            return X, y
        iid = data['id'].values
        return iid, X

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

    def csv_result(self, iid, result):
        df = pd.DataFrame({
            'id': iid,
            'is_driver_insurance': np.around(result, decimals=6),
        })

        if not os.path.isfile(self.result_path):
            df.to_csv(self.result_path, index=None, float_format='%0.6f')
        else:
            df.to_csv(self.result_path, index=None, float_format='%0.6f',\
                    mode='a', header=False)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    plt.ion()
    DriverInsurance.start()
