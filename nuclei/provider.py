# Nuclei data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import misc
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns


class Provider(object):
    
    def __init__(self):
        super(Provider, self).__init__()
        self.data_path = "data/nuclei/train"
        self.label_path = os.path.join(self.data_path, 'stage1_train_labels.csv')
        

    def load_label(self):
        self.lbl = pd.read_csv(self.label_path)

    def iter_data(self):

        for grp in glob.iglob(os.path.join(self.data_path, "?"*64)):
            img_grp = []
            target, img_id = None, None

            for path in glob.iglob(os.path.join(grp, 'images/*.png')):
                #print('++ [group] {}'.format(path))
                data = misc.imread(path)
                img_id = os.path.basename(path).split('.')[0]
                target = self.lbl[self.lbl['ImageId']==img_id]['EncodedPixels'].values
                img_grp.append(data)

            for path in glob.iglob(os.path.join(grp, 'masks/*.png')):
                #print('++ [found] {}'.format(path))
                data = misc.imread(path)
                img_grp.append(data)

            yield img_id, img_grp, target
                

    def get_ax(self, i):
        cur = i%(self.row*self.col)+1

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
        self.load_label()

        self.row, self.col = 5, 6

        self.axes = []
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
                    print(buf.shape)
                    print(buf[0])
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


def start():
    plt.ion()
    prov = Provider()
    prov.preprocess()


if __name__ == '__main__':
    #start()
    target = "66 35 322 36 578 38 834 39 1090 41 1346 42 1602 44 1858 45 2113 48 2367 52 2622 54 2878 55 3134 56 3390 57 3646 57 3902 58 4158 59 4414 59 4670 59 4926 60 5182 60 5438 61 5695 60 5951 60 6207 60 6463 60 6719 61 6975 61 7232 61 7488 61 7744 61 8000 61 8256 61 8513 60 8769 60 9026 59 9282 59 9539 58 9795 58 10051 58 10307 58 10564 57 10820 57 11076 57 11332 56 11588 56 11845 54 12102 52 12359 51 12616 50 12872 50 13129 48 13386 47 13643 45 13900 42 14157 40 14414 39 14671 37 14929 34 15186 30 15443 24 15700 22 15962 6 15972 5"
    target = "74746 7 74998 11 75253 12 75507 14 75762 15 76016 17 76271 18 76526 19 76781 20 77036 21 77291 22 77546 23 77802 23 78058 23 78313 24 78569 24 78825 24 79081 24 79336 25 79592 25 79848 25 80104 25 80361 24 80617 24 80874 23 81131 22 81388 21 81644 21 81901 20"
    plot_target(target, (256, 320, 4), None)
    plt.show()

