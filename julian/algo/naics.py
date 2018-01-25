# Model handler for NAICS
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import numpy as np
import pandas as pd
from julian.algo.model import Inference
from julian.algo.config import get_config


class Naics(Inference):

    def __init__(self):
        super(Naics, self).__init__()
        self.setup_s3cli()
        self.setup_model()
        self.d3table = self.load_d3table()
        self.class_map = list(set(self.d3table['code']))

    def load_d3table(self):
        cfg = get_config()
        cfg = cfg.naics
        path = os.path.join(cfg['data_base'], cfg['data']['d3_path'])
        return pd.read_csv(path,
                header=0, delimiter="#", dtype={"code":np.int})

    def level_decode(self, index):
        iid = self.class_map[index]
        code = self.d3table[self.d3table["code"] == iid].values
        code = np.squeeze(code).tolist()
        return iid, code[0], code[1]

    def decode(self, pred):
        ret = []
        pred = np.squeeze(pred).tolist()

        if not isinstance(pred, list):
            pred = [pred]

        for p in pred:
            iid, code, name = self.level_decode(p)
            ret.append({
                'iid': iid,
                'code': code,
                'name': name,
                })

        return ret
