# Prediction consumer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
import multiprocessing as mp
import ujson
import numpy as np
import pandas as pd
from julian.output.consumer import Prediction as PC
from julian.common.utils import raise_if_not_found
from julian.common.config import get_config
from julian.common.topic import Topic
from src.dynamodb.common.shared import DataType
from src.dynamodb.tables.journal_article import JournalArticle
from src.dynamodb.tables.organization import OrganizationTable


class Article(PC):
    """Prediction consumer for article"""
    def __init__(self, **kwargs):
        super(Article, self).__init__(topics=(Topic.PREDICT_TECH,),  **kwargs)
        self.cip_map_path = 'data/springer/springer_second.json'
        self.load_cip_map()
        self.cnt = 0

    def load_cip_map(self):
        with open(self.cip_map_path) as fd:
            self.cip_map = ujson.load(fd)

    def get_cip(self, p):
        return self.cip_map.get(p.replace('\\/', '/').replace(',', ''))

    def convert(self, **kwargs):
        for gid, sty, predict in zip(
            kwargs.get('global_id'),\
            kwargs.get('slice_type'),\
            kwargs.get('predict')):
            # each predict in structure [iid, l1, l2]
            gid = gid.decode()
            sty = sty.decode()
            predict = predict[2].decode()
            cip = self.get_cip(predict)

            if cip:
                cip = list(cip.keys())[0]
                yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'cip': cip,
                }
    def send(self, **kwargs):
        self.cnt += 1
        gid = kwargs.get('global_id')
        cip = kwargs['cip']
        # TODO REMOVE LATER
        print('++ [updated] {} {}'.format(gid, cip), self.cnt)
        return
        table = JournalArticle.rebuild(global_id=gid)
        if table.techdomain:
            table.techdomain[0]['techdomian_id'] = str(cip)
        else:
            table.techdomain = [{"techdomian_id":str(cip),}]
        table.save()
        print('++ [updated] {} {}'.format(gid, cip))


class Org(PC):
    """Prediction consumer for organization"""
    def __init__(self, **kwargs):
        super(Org, self).__init__(topics=(Topic.PREDICT_NAICS,), **kwargs)
        self.d3_path = "data/naics/codes_3digits.csv"
        self.d6_path = "data/naics/codes_6digits.csv"
        self.load_naics_tables()
        self.cnt = 0

    def load_naics_tables(self):
        raise_if_not_found(self.d3_path)
        raise_if_not_found(self.d6_path)
        self.d3_table = pd.read_csv(self.d3_path, header=0, delimiter='#', \
                dtype={'code': np.str})
        self.d6_table = pd.read_csv(self.d6_path, header=0, delimiter='#', \
                dtype={'code': np.str})

    def get_d6_code(self, in_id):
        # TODO Single code
        codes = self.d6_table[\
            self.d6_table['code'].str[:3]==str(in_id)].values.tolist()
        return codes[np.random.randint(len(codes))][0] if codes else None

    def convert(self, **kwargs):
        """Internal conversion"""
        for gid, sty, predict in zip(
            kwargs.get('global_id'),\
            kwargs.get('slice_type'),\
            kwargs.get('predict')):
            # each predict in structure [iid, d3, name]
            gid = gid.decode()
            sty = sty.decode()
            predict = str(predict[1])
            d6_code = self.get_d6_code(predict)

            if d6_code:
                yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'd6code': str(d6_code),
                }

    def send(self, **kwargs):
        self.cnt += 1
        gid = kwargs.get('global_id')
        d6_code = kwargs['d6code']
        # TODO REMOVE LATER
        print('++ [updated] {} {}'.format(gid, d6_code), self.cnt)
        return
        table = OrganizationTable.rebuild(global_id=gid)
        if table.industry_classifications:
            table.industry_classifications[0]['source_unique'] = str(d6_code)
        else:
            table.industry_classifications = [{'source_unique':str(d6_code),}]
        table.save()
        print('++ [updated] {} {}'.format(gid, d6_code))


def run_async(hdr_name):
    try:
        hdr = globals().get(hdr_name)()
        _ = list(hdr.run_async())
    except KeyboardInterrupt:
        print("++ [terminate] {}".format(hdr_name))


def start():
    pool = []
    try:
        for hdr_name in get_config().consumers.split(','):
            pool.append(mp.Process(target=run_async,
                                    args=(hdr_name,),
                                    name=hdr_name))
        list(map(lambda p: p.start(), pool))
    except KeyboardInterrupt:
        list(map(lambda p: p.join(), pool))
        print("++ [terminate]")


if __name__ == '__main__':
    start()
