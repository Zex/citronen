# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import ujson
import numpy as np
import pandas as pd


class Iceberg(object):

    def __init__(self):
        super(Iceberg, self).__init__()
    
    def load_data(self, path):
        if not os.path.isfile(path):
            return None
        data = pd.read_json(path)
        return data
    
    def preprocess(self):
        path = "data/iceberg/train.json"
        data = self.load_data(path)
        self.axes = []
        self.fig = plt.figure(figsize=(20, 10), facecolor='black', edgecolor='black')
        self.fig.show()
        self.cur_i = 0

        data[data['inc_angle']=='na'] = 0.1
        data['inc_angle'] = data['inc_angle'].astype(np.float64)

        for i, one in data.iterrows():
            self.plot_one(one, i)
            if i == 1604: input(); sys.exit()

    def plot_one(self, one, i):
        img_band_1 = np.array(one['band_1']).reshape(75, 75)
        img_band_2 = np.array(one['band_2']).reshape(75, 75)
        comb_add = (img_band_1+img_band_2)
        comb = (img_band_1+img_band_2)*one['inc_angle']
       
        grp = 4
        self.plot_img(img_band_1, one['inc_angle'], one['is_iceberg'], one['id'], self.cur_i*grp)
        self.plot_img(img_band_2, one['inc_angle'], one['is_iceberg'], one['id'], self.cur_i*grp+1)
        self.plot_img(comb_add, 'comb_add', one['is_iceberg'], one['id'], self.cur_i*grp+2)
        self.plot_img(comb, 'comb', one['is_iceberg'], one['id'], self.cur_i*grp+3)

        self.cur_i += 1

    def plot_img(self, img, angle, is_iceberg, iid, i):
        row = 6; col = 12
        cur = i%(row*col)+1

        if len(self.axes) > cur:
            ax = self.axes[cur]
            ax.clear()
        else:
            ax = self.fig.add_subplot(row, col, i%(row*col)+1)
            self.axes.append(ax)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img)
        ax.annotate('angle {}'.format(angle), xy=(1,6))
        ax.annotate('iceberg {}'.format(is_iceberg), xy=(1,14))
        ax.annotate('id {}'.format(iid), xy=(1,23))
        plt.tight_layout(w_pad=0.1, h_pad=0.05, pad=0.1)
        self.fig.canvas.draw()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    plt.ion()
    ice = Iceberg()
    ice.preprocess()
