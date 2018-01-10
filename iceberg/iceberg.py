# Iceberg identifier
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


class Iceberg(object):

    def __init__(self, args):
        super(Iceberg, self).__init__()
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
        data = pd.read_json(path)
        return data
   
    def analyze(self):
        path = "data/iceberg/train.json"
        #path = "data/iceberg/test.json"
        #data = self.load_data(path)
        self.axes = []
        self.fig = plt.figure(figsize=(20, 10), facecolor='grey', edgecolor='black')
        #self.fig.show()
        self.cur_i = 0
        
        #data[data['inc_angle']=='na'] = 0.1
        #data['inc_angle'] = data['inc_angle'].astype(np.float64)

        #for i, one in enumerate(self.iload_data(path)):
        for one in data.iterrows():
            self.plot_one(one, i)
            if i == 1060: input(); sys.exit()

    def plot_one(self, one, i):
        img_band_1 = np.array(one['band_1']).reshape(75, 75)
        img_band_2 = np.array(one['band_2']).reshape(75, 75)
        one['inc_angle'] = 1.0 if one['inc_angle'] == 'na' else float(one['inc_angle'])
        comb_add = np.array(one['band_1'])+np.array(one['band_2'])
        comb_add = comb_add.reshape(75, 75)

        inc_angle = one['inc_angle']
        is_iceberg = -1 #one['is_iceberg']
        #comb = (img_band_1+img_band_2)*one['inc_angle']

        grp = 3
        self.plot_img(img_band_1, inc_angle, is_iceberg, one['id'], self.cur_i*grp)
        self.plot_img(img_band_2, inc_angle, is_iceberg, one['id'], self.cur_i*grp+1)
        self.plot_img(comb_add, 'comb_add', is_iceberg, one['id'], self.cur_i*grp+2)
        #self.plot_img(comb, 'comb', is_iceberg, one['id'], self.cur_i*grp+3)

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

    def load_data(self, path):
        if not os.path.isfile(path):
            return None
        data = pd.read_json(path)
        return data

    def iload_data(self, path):
        if not os.path.isfile(path):
            return None

        with open(path) as fd:
            one = {}
            band, buf, met_colon, list_on = [], '', False, False

            while True:
                if all([True if k in one else False \
                        for k in ('id', 'band_1', 'band_2', 'inc_angle')]):
                    yield one
                    one = {}
                    band, buf, met_colon, list_on = [], '', False, False

                c = fd.read(1).strip()
                if not c:
                    break
                if c == ':':
                    met_colon = True
                    continue
                if not met_colon:
                    continue
                if c == '[':
                    list_on = True
                    continue
                if c in '-'+'.'+string.digits+string.ascii_letters:
                   buf += c
                   continue
                if c == ']':
                    list_on = False
                    band.append(float(buf))

                    if len(one.get('band_1', [])) == 0:
                        one['band_1'] = np.array(band)
                        band, buf, met_colon, list_on = [], '', False, False
                        continue
                    if len(one.get('band_2', [])) == 0:
                        one['band_2'] = np.array(band)
                        band, buf, met_colon, list_on = [], '', False, False
                        continue
                if c == ',':
                    if list_on:
                        band.append(float(buf))
                        buf = ''
                        continue
                    if not one.get('id'):
                        one['id'] = buf
                        band, buf, met_colon, list_on = [], '', False, False
                        continue
                    if not one.get('inc_angle') and not list_on:
                        one['inc_angle'] = float(buf) if buf != 'na' else 1.0
                        band, buf, met_colon, list_on = [], '', False, False

    
    def preprocess(self):
        data = self.load_data(self.path)

        iid = data['id']
        band_1 = data['band_1']
        band_2 = data['band_2']
        data['inc_angle'] = data[data['inc_angle']=='na'] = '1.0'
        angle = data['inc_angle'].astype(np.float64)

        X = list(map(lambda l: (np.array(l[1][0])+np.array(l[1][1]).T), \
                enumerate(zip(band_1.values, band_2.values))))
        #X = list(map(lambda l: (np.array(l[1][0])+np.array(l[1][1])).T*l[1][2], \
        #        enumerate(zip(band_1.values, band_2.values, angle.values))))

        if self.mode in (Mode.TRAIN, Mode.EVAL):
            label = data['is_iceberg']
            y = label.values.reshape(len(label), 1)
            return X, y
        return iid.values, X

    def load_model(self):
        pass

    @classmethod
    def init(cls):
        parser = argparse.ArgumentParser(description="Iceberg Identifier")
        parser.add_argument('--train', action='store_true', default=False, help='train a model')
        parser.add_argument('--test', action='store_true', default=False, help='test a model')
        parser.add_argument('--eval', action='store_true', default=False, help='eval a model')
        parser.add_argument('--anal', action='store_true', default=False, help='analyse data')
        parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
        parser.add_argument('--load_model', action='store_true', default=False, help='load exist model')
        parser.add_argument('--model_dir', type=str, default='models/iceberg', help='model directory')
        return parser.parse_args()

    @classmethod
    def start(cls):
        args = cls.init()
        ice = cls(args)
    
        if args.train:
            ice.train()
        if args.test:
            ice.test()
        if args.eval:
            ice.eval()
        if args.anal:
            ice.analyze()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    plt.ion()
    Iceberg.start()
