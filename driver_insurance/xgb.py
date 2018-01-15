# DriverInsurance identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import sys
import os
sys.path.insert(0, os.getcwd())
import string
import glob
import ujson
import numpy as np
import pandas as pd
from datetime import datetime
import xgboost as xgb
from driver_insurance.driver_insurance import DriverInsurance, Mode


class Xgb(DriverInsurance):

    def __init__(self, args):
        super(Xgb, self).__init__(args)

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/driver_insurance/train.csv"
        if self.has_model:
            self.load_model()

        X, y = self.preprocess()
        for epoch in range(1, self.epochs+1):
            self.foreach_epoch(epoch, X, y)

    def test(self):
        self.mode = Mode.TEST
        self.path = "data/driver_insurance/test.csv"
        self.load_model()

        def pred(iid, X):
            res = self.model.predict(xgb.DMatrix(X))
            print('++ [pred] {}'.format(res))
            self.csv_result(iid, res)
            print('++ [result_path] {}'.format(self.result_path))

        scores = self.model.get_score()
        print('++ [feature_score] {}'.format(len(scores)))

        if os.path.isfile(self.result_path):
            os.remove(self.result_path)

        iid, X = self.preprocess()
        pred(iid, X)

    def eval(self):
        self.mode = Mode.EVAL
        self.path = "data/driver_insurance/train.csv"
        self.load_model()

        scores = self.model.get_score()
        print('++ [feature_score] {}'.format(len(scores)))

        X, y = self.preprocess()
        dtrain = xgb.DMatrix(X, y)

        res = self.model.eval(dtrain)
        print('++ [eval] {}'.format(res))

    def foreach_epoch(self, epoch, X, y):
        params = {
            'learning_rate': self.lr,
            'update':'refresh',
#           'process_type': 'update',
            'refresh_leaf': True,
            'reg_lambda': 0.23,
            'max_depth': 6,
#            'reg_alpha': 3,
            'silent': False,
            'n_jobs': 3,
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
                num_boost_round=7,
                callbacks=[self.callback_iter])

        pred = self.model.predict(xgb.DMatrix(X))
        print("++ [epoch-{}] pred:{}\nlbl:{}".format(epoch, [x for x in np.round(pred, 4)], [x for x in y.T]))
        self.model_path = os.path.join(self.model_dir, 'driver_insurance-{}.xgb'.format(self.last_epoch+epoch))
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
        if str(env.evaluation_result_list[0][1]) == '0.0':
            self.error_stop_cnt += 1
        if self.error_stop_cnt == 7:
            sys.exit()

    def load_model(self):
        mod = glob.glob('{}/driver_insurance-*.xgb'.format(self.model_dir))
        mod = sorted(mod, key=lambda p:int(os.path.basename(p).split('.')[0].split('-')[1]))

        if mod:
            self.model = xgb.Booster()
            mod = mod[-1]
            print('++ [info] model:{}'.format(mod))
            self.model.load_model(mod)
            self.last_epoch = int(os.path.basename(mod).split('.')[0].split('-')[1])

    
if __name__ == '__main__':
    Xgb.start()
