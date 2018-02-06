# Nuclei data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import glob
import numpy as np
import pandas as pd
from functools import reduce
import argparse
from scipy import misc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.misc import imresize
#%matplotlib inline

import seaborn as sns

def foreach_target(target_grp, img_grp):
    target_grp = [int(t) for t in target_grp.split()]
    if target_grp[-2] + target_grp[-1] > \
            img_grp[0].shape[0] * img_grp[0].shape[1]:
        target_grp = target_grp[:-2]
    return target_grp

class Provider(object):
    
    def __init__(self, args=None):
        super(Provider, self).__init__()
        self.data_path = "data/nuclei/train"
        self.label_path = os.path.join(self.data_path, 'stage1_train_labels.csv')
        self.width = 512
        self.height = 512
        self.channel = 4
        self.batch_size = args.batch_size if args else 1
        self.load_label()

    def load_label(self):
        self.lbl = pd.read_csv(self.label_path)


    def iter_data(self):

        for grp in glob.iglob(os.path.join(self.data_path, "?"*64)):
            img_grp = []
            target, img_id = None, None

            for path in glob.iglob(os.path.join(grp, 'images/*.png')):
                #print('++ [group] {}'.format(path))
                data = misc.imread(path)
                data = imresize(data, (self.height, self.width, self.channel))
                img_id = os.path.basename(path).split('.')[0]
                target = self.lbl[self.lbl['ImageId']==img_id]['EncodedPixels'].values
                img_grp.append(data)

            for path in glob.iglob(os.path.join(grp, 'masks/*.png')):
                #print('++ [found] {}'.format(path))
                data = misc.imread(path)
                data = imresize(data, (self.height, self.width, self.channel))
                img_grp.append(data)

            list(map(lambda t: foreach_target(t, img_grp), target))
            yield img_id, img_grp, target
                

    def get_ax(self, i):
        cur = i%(self.row*self.col)+1
        if not getattr(self, 'axes', None):
            self.axes = []

        if len(self.axes) > cur:
            ax = self.axes[cur]
            ax.clear()
        else:
            ax = self.fig.add_subplot(self.row, self.col, i%(self.row*self.col)+1)
            self.axes.append(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.clear()
        return ax

    def preprocess(self):
        self.row, self.col = 5, 6
        self.fig = plt.figure(figsize=(10, 8), facecolor='grey', edgecolor='black')
        self.fig.show()

        img = None

        for gi, (img_id, img_grp, target) in enumerate(self.iter_data()):
            if gi < 30:
                continue
            for i, data in enumerate(img_grp):
                print('++ [shape] {} {}'.format(data.shape, target[i-1] if i > 0 else None))
                if i > 0:
                    ax = self.get_ax(i)
                    plot_target(target[i-1], img.shape, ax)
                if len(data.shape) == 3:
                    buf = data.reshape([data.shape[2], data.shape[0]*data.shape[1]])
                    #[print(row) for row in buf[0]]
                if i == 0:
                    img = data
                    for d in range(4):
                        ax = self.get_ax(i+d)
                        ax.imshow(data[:,:,d])
                        ax.annotate('img {}'.format(''.join(list(img_id)[-4:])), xy=(5,10))

                if i != 0:
                    ax = self.get_ax(i+2)
                    ax.imshow(data)
                    ax.annotate('mask {}'.format(''), xy=(5,10))
                    sub = self.get_ax(i+3)
                    sub.imshow(img[:,:,0]*data)
                    sub.annotate('filter {}'.format(''), xy=(5,10))

                plt.tight_layout(w_pad=0.1, h_pad=0.05, pad=0.1)
                self.fig.canvas.draw()
            break

    def plot_Xmask(self, X, mask):
        self.row, self.col = 1, 2
        if not getattr(self, 'fig', None):
            self.fig = plt.figure(figsize=(10, 8), facecolor='grey', edgecolor='black')
        ax = self.get_ax(0)
        ax.imshow(X)
        ax = self.get_ax(1)
        ax.imshow(mask)

    def gen_data(self):
        X_batch, y_batch = [], []
        total_cls = []

        for gi, (img_id, img_grp, target) in enumerate(self.iter_data()):
            #if gi < 50:
            #    continue
            X = img_grp[0]
            masks = img_grp[1:]
            mask = np.zeros(masks[0].shape)
            mask = reduce(lambda mask, m: mask+m, masks)
            
            #self.plot_Xmask(X, mask)
            #yield X, mask
            if len(X_batch) < self.batch_size:
                X_batch.append(X)
                y_batch.append(mask)
                total_cls.append(len(masks))
                continue
            total_cls = np.array(total_cls).reshape(len(total_cls), 1) 
            yield np.array(X_batch).astype(np.int32), np.array(y_batch).astype(np.int32), total_cls
            X_batch.clear(); y_batch.clear()
            total_cls = []


def plot_target(target, shape, ax=None):
    if isinstance(target, str):
        seq = target.split()
    else:
        seq = target

    start = [int(t) for i, t in enumerate(seq) if i % 2 == 0] 
    count = [int(t) for i, t in enumerate(seq) if i % 2 != 0]
    mask = np.zeros(shape[1]*shape[0])

    for s, c in zip(start, count):
        for ind in range(s, s+c+1):
            if ind >= mask.shape[0]:
                continue
            mask[ind] = 255

    mask = mask.reshape(shape[1], shape[0])
    encode = encode_mask(mask) 
    #print(target == encode)

    if ax:
        ax.imshow(mask)
        #ax.scatter(start, count, linewidths=2, s=1)
     
    return mask


def encode_mask(mask):
    mask = mask.flatten()
    nonzero = mask.nonzero()[0]
    bucket, target = [], []

    for cell in nonzero:
        if not bucket:
            bucket.append(cell)
            continue
        if cell-1 not in bucket:
            target.extend([str(bucket[0]), str(len(bucket)-1)])
            bucket.clear()
        bucket.append(cell)

    target.extend([str(bucket[0]), str(len(bucket)-1)])

    return ' '.join(target)

def init():
    parser = argparse.ArgumentParser(description="Price estimation")
    parser.add_argument('--train', action='store_true', default=False, help='train a model')
    parser.add_argument('--test', action='store_true', default=False, help='test a model')
    parser.add_argument('--eval', action='store_true', default=False, help='eval a model')
    parser.add_argument('--anal', action='store_true', default=False, help='analyse data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000000, help='total epochs')
    parser.add_argument('--load_model', action='store_true', default=False, help='load exist model')
    parser.add_argument('--model_dir', type=str, default='models/price', help='model directory')
    parser.add_argument('--lr', type=float, default=1e-10, help='initial learning rate')
    parser.add_argument('--summ_intv', type=int, default=1000, help='summary interval')
    parser.add_argument('--init_step', type=int, default=1, help='initial step')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
    return parser.parse_args()

def start():
    plt.ion()
    prov = Provider()
    prov.preprocess()


if __name__ == '__main__':
    #start()
    target = "66 35 322 36 578 38 834 39 1090 41 1346 42 1602 44 1858 45 2113 48 2367 52 2622 54 2878 55 3134 56 3390 57 3646 57 3902 58 4158 59 4414 59 4670 59 4926 60 5182 60 5438 61 5695 60 5951 60 6207 60 6463 60 6719 61 6975 61 7232 61 7488 61 7744 61 8000 61 8256 61 8513 60 8769 60 9026 59 9282 59 9539 58 9795 58 10051 58 10307 58 10564 57 10820 57 11076 57 11332 56 11588 56 11845 54 12102 52 12359 51 12616 50 12872 50 13129 48 13386 47 13643 45 13900 42 14157 40 14414 39 14671 37 14929 34 15186 30 15443 24 15700 22 15962 6 15972 5"
    target = "74746 7 74998 11 75253 12 75507 14 75762 15 76016 17 76271 18 76526 19 76781 20 77036 21 77291 22 77546 23 77802 23 78058 23 78313 24 78569 24 78825 24 79081 24 79336 25 79592 25 79848 25 80104 25 80361 24 80617 24 80874 23 81131 22 81388 21 81644 21 81901 20"
#    plot_target(target, (256, 320, 4), None)
    prov = Provider()

    for X, y, n_cls in prov.gen_data():
        print(X[0], y[0], n_cls)
    plt.show()
