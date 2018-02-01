# Nuclei data provider
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import use as muse
muse("TkAgg")
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
            target = None

            for path in glob.iglob(os.path.join(grp, 'images/*.png')):
                print('++ [group] {}'.format(path))
                data = misc.imread(path)
                img_id = os.path.basename(path).split('.')[0]
                target = self.lbl[self.lbl['ImageId']==img_id]['EncodedPixels'].values
                img_grp.append(data)

            for path in glob.iglob(os.path.join(grp, 'masks/*.png')):
                print('++ [found] {}'.format(path))
                data = misc.imread(path)
                img_grp.append(data)

            yield img_grp, target
                

    def get_ax(self, i):
        cur = i%(self.row*self.col)+1

        if len(self.axes) > cur:
            ax = self.axes[cur]
            ax.clear()
        else:
            ax = self.fig.add_subplot(self.row, self.col, i%(self.row*self.col)+1)
            self.axes.append(ax)
        return ax

    def preprocess(self):
        self.load_label()

        self.row, self.col = 1, 20

        self.axes = []
        self.fig = plt.figure(figsize=(15, 6), facecolor='grey', edgecolor='black')
        self.fig.show()

        for img_grp, target in self.iter_data():
            for i, data in enumerate(img_grp):
                print('++ [shape] {} {}'.format(data.shape,
                    target[i-1] if i > 0 else None))
                ax = self.get_ax(i)
                ax.imshow(data)
                plt.tight_layout(w_pad=0.1, h_pad=0.05, pad=0.1)
                self.fig.canvas.draw()


def start():
    plt.ion()
    prov = Provider()
    prov.preprocess()


if __name__ == '__main__':
    start()
