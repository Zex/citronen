# DriverInsurance identifier
# Author: Zex Li <top_zlynch@yahoo.com>
import sys
import os
sys.path.insert(0, os.getcwd())
import glob
import pickle
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from driver_insurance.driver_insurance import DriverInsurance, Mode


class RT(DriverInsurance):

    def __init__(self, args):
        super(RT, self).__init__(args)

    def train(self):
        self.mode = Mode.TRAIN
        self.path = "data/driver_insurance/train.csv"
        X, y = self.preprocess()
        y = np.squeeze(y)

        if self.has_model:
            self.load_model()
        else:
            #self.model = RandomForestClassifier(n_estimators=1024, max_depth=10, verbose=2, n_jobs=30)
            self.model = RandomForestRegressor(n_estimators=10, max_depth=10, verbose=1, n_jobs=40, criterion='mae')

        self.model = self.model.fit(X, y)
        self.save_model()
        score = self.model.score(X, y)
        print('++ [train] score:{}'.format(score))
        print('++ [train] feat:{}'.format(self.model.n_features_))
        print('++ [train] importance:{}'.format(self.model.feature_importances_))

    def test(self):
        self.mode = Mode.TEST
        self.path = "data/driver_insurance/test.csv"

        self.load_model()
        iid, X = self.preprocess()

        res = self.model.predict(X)
        self.csv_result(iid, res)
        print('++ [result_path] {}'.format(self.result_path))

    def save_model(self):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        output = os.path.join(self.model_dir, \
                "{}.pickle".format(datetime.now().strftime("%Y%m%d%H%M")))

        with open(output, 'wb') as fd:
            pickle.dump(self.model, fd)

    def load_model(self):
        output = glob.glob('{}/*pickle'.format(self.model_dir)) 
        if not output:
            return None

        sorted(output, key=lambda f:os.stat(f).st_mtime)
        output = output[-1]
        print("++ [test] model:{}".format(output))

        with open(output, 'rb') as fd:
            self.model = pickle.load(fd)

if __name__ == '__main__':
    RT.start()
