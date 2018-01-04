# Iceberg identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import sys
import os
sys.path.insert(0, os.getcwd())
import sys
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
    
    def preprocess(self):
        data = self.load_data(self.path)

        iid = data['id']
        band_1 = data['band_1']
        band_2 = data['band_2']

        #X = np.array(list(map(lambda val:np.array(val), band_1.values)))
        #X = list(map(lambda l: l[1][0]+l[1][1], enumerate(zip(band_1.values, band_2.values))))
        X = list(map(lambda l: np.array(l[1][0])+np.array(l[1][1]), enumerate(zip(band_1.values, band_2.values))))
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
            self.model.load_model(mod[-1])

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

        scores = self.model.get_score()
        print('++ [feature_score] {}'.format(len(scores)))

        iid, X = self.preprocess()
        pred = self.model.predict(xgb.DMatrix(X))
        print('++ [pred] {}'.format(pred))
        self.csv_result(iid, pred)

    def csv_result(self, iid, result):
        df = pd.DataFrame({
            'id': iid,
            'is_iceberg': np.around(result, decimals=1),
        })

        df.to_csv(self.result_path, index=None, float_format='%0.1f')

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
    ice.train()
    #ice.test()
