#

import ujson
import os
import pandas as pd
from matplotlib import pyplot as plt


class Iceberg(object):

    
    def load_data(self, path):
        if not os.path.isfile(path):
            return None
        #with open(path) as fd:
        #    data = ujson.load(fd)
        data = pd.read_json(path)
        return data
    
    def preprocess(self):
        path = "data/iceberg/train.json"
        data = self.load_data(path)
        print(len(data))
        one = data.iloc[0]
        print(one)



if __name__ == '__main__':
    ice = Iceberg()
    ice.preprocess()
