# XGB
#
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from price.provider import *


class Price(object):

    def __init__(self):
        super(Price, self).__init__()
        self.lr = 1e-4
        self.model = None
        self.cfg = Config()
        self.cfg.evel_result_path = "data/price/eval_xgb_{}.csv".format(\
                    datetime.now().strftime("%y%m%d%H%M"))

    def foreach_batch(self, X, y):
        params = {
              'learning_rate': self.lr,
              'update':'refresh',
              'refresh_leaf': True,
              'reg_lambda': 0.1,
              'max_depth': 10,
#              'reg_alpha': 3,
              'silent': False,
              'n_jobs': 2,
              'objective': 'reg:linear',
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
        try:
            print("++ [iter {} {}/{}] eval:{}".format(\
                env.begin_iteration, env.iteration, \
                env.end_iteration, env.evaluation_result_list))

            with open(self.cfg.eval_result_path, 'a') as fd:
                fd.write(ujson.dumps({
                    datetime.now().timestamp(): {
                        'step': env.iteration,
                        'train_err': env.evaluation_result_list[0][1],
                        }})+'\n')
            if str(env.evaluation_result_list[0][1]) == '0.0':
                self.error_stop_cnt += 1
            if self.error_stop_cnt == 7:
                sys.exit()
        except Exception as ex:
              print("-- [error] {}".format(ex))
       
            
    def train(self):
        for X, y in preprocess(self.cfg):
            print('++ [info] {}'.format(X.shape))
            self.foreach_batch(X, y)
            score = self.score(X, y)
            print('++ [score] {}'.format(score))

    def inner_test(self, step):
        def foreach_chunk(iid, X):
            pred = self.model.predict(X)
            df = pd.DataFrame({
                    'test_id': iid,
                    'price': pred,
                })
            to_csv(df, cfg.result_path)

        cfg = Config()
        cfg.path = "data/price/test.tsv"
        cfg.need_shuffle = False
        cfg.mode = Mode.TEST
        cfg.result_path = "data/price/pred_xgb_{}_{}.csv".format(\
                    step,
                    datetime.now().strftime("%y%m%d%H%M"))
    
        gen = preprocess(cfg)
        list(map(lambda iid, X: foreach_chunk(iid, X), gen))


def start():
    obj = Price()
    obj.train()


if __name__ == '__main__':
    start()
