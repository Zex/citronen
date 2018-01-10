# Model handler for NAICS
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import pickle
import ujson
import numpy as np
from julian.algo.model import Interfere
from julian.algo.config import get_config


class Springer(Interfere):

    def __init__(self):
        super(Springer, self).__init__()
        self.setup_s3cli()
        self.setup_model()
        self.load_table()
        self.class_map = list(set(self.l2table.values()))

    def load_table(self):
        cfg = get_config()
        cfg = cfg.springer

        path = os.path.join(cfg['data_base'], cfg['data']['l2_path'])
        if os.path.isfile(path):
            with open(path, 'rb') as fd:
                self.l2table = pickle.load(fd)

        path = os.path.join(cfg['data_base'], cfg['data']['l1_path'])
        if os.path.isfile(path):
            with open(path, 'rb') as fd:
                self.l1table = pickle.load(fd)

        path = os.path.join(cfg['data_base'], cfg['data']['cip_map'])
        with open(path) as fd:
            self.cip_map = ujson.load(fd)

    def level_decode(self, index):

        iid, l1name, l2name = None, None, None

        if self.l2table:
            if not self.class_map:
                self.class_map = list(set(self.l2table.values()))
            iid = self.class_map[index]
            l2name = dict(map(reversed, self.l2table.items())).get(iid)
            if self.l1table:
                l1name = dict(map(reversed, self.l1table.items())).get(iid//0x1000*0x1000)
        return iid, l1name, l2name

    def get_cip(self, p):
        return self.cip_map.get(p.replace('\\/', '/').replace(',', ''))

    def decode(self, pred):
        pred = np.squeeze(pred).tolist()
        ret = []

        if not isinstance(pred, list):
            pred = [pred]

        for p in pred:
            iid, l1, l2 = self.level_decode(p)
            ret.append({
                'iid': iid,
                'l1': l1,
                'l2': l2,
                })

        return ret


def selftest():
    from datetime import datetime
    start = datetime.now()
    obj = Springer()
    res = obj.interfere([
        'GOOTEN MORGEN',
        'ICH BIN FLEXHEN EINEN HERONIGEN',
        'WUSHIBIDIE ZWEIZICH',
        ])
    print('++ [result] {}'.format(res))
    print('++ [elapsed] {}'.format(datetime.now()-start))

if __name__ == '__main__':
    selftest()
