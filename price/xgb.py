# Train with XGB
#
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from price.provider import *


class Price(object):

    def __init__(self):
        super(Price, self).__init__()

    def _build_model(self):
        self.model = xgb.XGBRegressor(
            max_depth=8,
            learning_rate=1e-4,
            n_jobs=4,
            silent=False,
        )

    def train(self):
        global BATCH_SIZE
        BATCH_SIZE = 1482536
        BATCH_SIZE = 128

        self._build_model()

        for X, y in preprocess():
            self.model.fit(X, y, verbose=True)
            score = self.score(X, y)
            print('++ [score] {}'.format(score))

    def inner_test(self, sess, step):
        global path

        def foreach_chunk(iid, X):
            pred = self.model.predict(X)
            df = pd.DataFrame({
                    'test_id': iid,
                    'price': pred,
                })

            if not os.path.isfile(result_path):
                df.to_csv(result_path, index=None, float_format='%0.6f')
            else:
                df.to_csv(result_path, index=None, float_format='%0.6f', header=False, mode='a')

        prev_path = path
        path = "data/price/test.tsv"
        result_path = "data/price/pred_xgb_{}_{}.csv".format(\
                    step,
                    datetime.now().strftime("%y%m%d%H%M"))
    
        gen = preprocess(need_shuffle=False, mode='TEST')
        [foreach_chunk(iid, X) for iid, X in gen]
        path = prev_path


def start():
    obj = Price()
    obj.train()


if __name__ == '__main__':
    start()
