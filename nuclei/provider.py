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
%matplotlib inline

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


def start():
    plt.ion()
    prov = Provider()
    prov.preprocess()


if __name__ == '__main__':
    start()

