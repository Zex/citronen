# Iceberg classification
# Author: Zex Li <top_zlynch@yahoo.com>

import ujson
import os
import sys
import numpy as np
import pandas as pd
from iceberg.iceberg import Iceberg


class Xgb(Iceberg):

    def __init__(self):
        super(Xgb, self).__init__()
    
    def load_data(self, path):
        if not os.path.isfile(path):
            return None
        data = pd.read_json(path)
        return data
    
    def preprocess(self):
        path = "data/iceberg/train.json"
        data = self.load_data(path)
        band_1 = data['band_1']
        band_2 = data['band_2']
        label = data['is_iceberg']
        return band_1, label

    def train(self):
        train_X, train_y = self.preprocess()
        xgb_train = xgb.DMatrix(train_X, label=train_y)
        print(xgb_train)


if __name__ == '__main__':
    ice = Xgb()
    ice.train()
