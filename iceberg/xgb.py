# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import sys
import os
sys.path.insert(0, os.getcwd())
import sys
import string
import glob
import ujson
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from iceberg.iceberg import Iceberg
#from matplotlib import pyplot as plt


class Mode(object):
    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'

class Xgb(Iceberg):

    def __init__(self):
        super(Xgb, self).__init__()
        self.lr = 1e-3
        self.batch_size = 100
        self.steps = 1000
        self.model = None
        self.model_dir = 'models/iceberg'
        self.eval_result_path = os.path.join(self.model_dir, 'logs', 'eval.json')

        base = os.path.dirname(self.eval_result_path)
        if not os.path.isdir(base):
            os.makedirs(base)
    
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

                if all([True if k in one else False \
                        for k in ('id', 'band_1', 'band_2', 'inc_angle')]):
                    yield one
                    one = {}
                    band, buf, met_colon, list_on = [], '', False, False
    
    def preprocess(self):
        data = self.load_data(self.path)

        iid = data['id']
        band_1 = data['band_1']
        band_2 = data['band_2']
        data['inc_angle'] = data[data['inc_angle']=='na'] = '1.0'
        angle = data['inc_angle'].astype(np.float64)

        #X = list(map(lambda l: l[1][0]+l[1][1], \
        #        enumerate(zip(band_1.values, band_2.values))))
        X = list(map(lambda l: np.array(l[1][0])+np.array(l[1][1])*l[1][2], \
                enumerate(zip(band_1.values, band_2.values, angle.values))))
        X = np.array(X)

        if self.mode in (Mode.TRAIN, Mode.EVAL):
            label = data['is_iceberg']
            y = label.values.reshape(len(label), 1)
            return X, y
        return iid.values, X

    def load_model(self):
        mod = glob.glob('{}/*.xgb'.format(self.model_dir))
        if mod:
            self.model = xgb.Booster()
            mod = mod[-1]
            print('++ [info] model:{}'.format(mod))
            self.model.load_model(mod)

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"
        self.load_model()

        for epoch in range(1, self.steps+1):
            self.foreach_epoch(epoch)

    def test(self):
        self.mode = Mode.TEST
        self.path = "data/iceberg/test.json"
        self.result_path = "data/iceberg/pred.csv"
        self.load_model()

        def pred(iid, X):
            res = self.model.predict(xgb.DMatrix(X))
            print('++ [pred] {}'.format(res))
            self.csv_result(iid, res)

        scores = self.model.get_score()
        print('++ [feature_score] {}'.format(len(scores)))

        if os.path.isfile(self.result_path):
            os.remove(self.result_path)

        #iid, X = self.preprocess()
        #pred(iid, X)

        for one in self.iload_data(self.path):
            X = np.array([one.get('band_1')+one.get('band_2')])*one.get('inc_angle')
            pred(one.get('id'), X) 

    def csv_result(self, iid, result):
        df = pd.DataFrame({
            'id': iid,
            'is_iceberg': np.around(result, decimals=4),
        })

        if not os.path.isfile(self.result_path):
            df.to_csv(self.result_path, index=None, float_format='%0.4f')
        else:
            df.to_csv(self.result_path, index=None, float_format='%0.4f',\
                    mode='a', header=False)

    def eval(self):
        self.mode = Mode.EVAL
        self.load_model()

        scores = self.model.get_score()
        print('++ [feature_score] {}'.format(len(scores)))

        X, y = self.preprocess()
        dtrain = xgb.DMatrix(X, y)

        res = self.model.eval(dtrain)
        print('++ [eval] {}'.format(res))

        plt.scatter(range(len(pred)), pred, color='r', s=5)
        plt.scatter(range(len(pred)), y, color='b', s=5)
        plt.scatter(range(len(pred)), \
            np.array(np.squeeze(y).round())-np.array(pred.astype(np.float)),\
            color='g', s=5)
        plt.show()

    def foreach_epoch(self, epoch):
        X, y = self.preprocess()
        params = {
            'learning_rate': self.lr,
            'update':'refresh',
            #'process_type': 'update',
            'refresh_leaf': True,
            'reg_lambda': 1,
            #'reg_alpha': 3,
            'silent': False,
            'objective': 'binary:logistic',
            'eval_metrics': 'logloss',
        } 
        eval_res = {}
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(params,
                dtrain=dtrain,
                xgb_model=self.model,
                evals=[(dtrain, 'train')],
                evals_result=eval_res,
                num_boost_round=20,
                callbacks=[self.callback_iter])

        pred = self.model.predict(xgb.DMatrix(X))
        print("++ [epoch-{}] pred:{}\nlbl:{}".format(epoch, np.round(pred, 4), y.T))
        self.model_path = os.path.join(self.model_dir, 'iceberg-{}.xgb'.format(epoch))
        self.model.save_model(self.model_path)

    def callback_iter(self, env):
        print("++ [iter {} {}/{}] eval:{}".format(\
            env.begin_iteration, env.iteration, \
            env.end_iteration, env.evaluation_result_list))
        with open(self.eval_result_path, 'a') as fd:
            fd.write(ujson.dumps({
                datetime.now().timestamp(): {
                    'step': env.iteration,
                    'train_err': env.evaluation_result_list[0][1],
                    }})+'\n')


if __name__ == '__main__':
    ice = Xgb()
    #ice.train()
    ice.test()
