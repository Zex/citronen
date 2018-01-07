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
from iceberg.iceberg import Iceberg, Mode
from matplotlib import pyplot as plt



class Xgb(Iceberg):

    def __init__(self, args):
        super(Xgb, self).__init__(args)
        self.lr = 1e-2
        self.batch_size = 100
        self.steps = 1000
        self.model = None
        self.model_dir = args.model_dir #'models/iceberg'
        self.eval_result_path = os.path.join(self.model_dir, 'logs', 'eval.json')
        self.has_model = args.load_model
        self.error_stop_cnt = 0
        self.last_epoch = 0

        base = os.path.dirname(self.eval_result_path)
        if not os.path.isdir(base):
            os.makedirs(base)

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/iceberg/train.json"
        if self.has_model:
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

        ax = xgb.plot_importance(self.model)
        plt.savefig("data/iceberg/feature_importance_plot.png")

        ax = xgb.plot_tree(self.model)
        plt.savefig("data/iceberg/feature_tree_plot.png")

        if os.path.isfile(self.result_path):
            os.remove(self.result_path)

        #iid, X = self.preprocess()
        #pred(iid, X)

        for one in self.iload_data(self.path):
            X = np.array([one.get('band_1')+one.get('band_2')])#*one.get('inc_angle')
            pred(one.get('id'), X) 

    def csv_result(self, iid, result):
        df = pd.DataFrame({
            'id': iid,
            'is_iceberg': np.around(result, decimals=6),
        })

        if not os.path.isfile(self.result_path):
            df.to_csv(self.result_path, index=None, float_format='%0.6f')
        else:
            df.to_csv(self.result_path, index=None, float_format='%0.6f',\
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
            'reg_lambda': 0.1,
            'max_depth': 8,
            #'reg_alpha': 3,
            'silent': False,
            'n_jobs': 2,
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
        self.model_path = os.path.join(self.model_dir, 'iceberg-{}.xgb'.format(self.last_epoch+epoch))
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
        mod = glob.glob('{}/*.xgb'.format(self.model_dir))
        if mod:
            self.model = xgb.Booster()
            mod = mod[-1]
            print('++ [info] model:{}'.format(mod))
            self.model.load_model(mod)
            self.last_epoch = int(os.path.basename(mod).split('.')[0].split('-')[1])

    
if __name__ == '__main__':
    Xgb.start()
