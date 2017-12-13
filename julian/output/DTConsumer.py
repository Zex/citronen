# Prediction consumer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
import multiprocessing as mp
import ujson
from julian.input.consumer import Prediction as PC
from src.dynamodb.common.shared import DataType
from src.dynamodb.tables import JournalArticle


class Article(PC):
    """Prediction consumer for article"""
    def __init__(self, **kwargs):
        super(Article, self).__init__(**kwargs)
        self.cip_map_path = 'springer_second.json'
        self.load_cip_map()

    def load_cip_map(self):
        with open(self.cip_map_path) as fd:
            self.cip_map = ujson.load(fd)

    def get_cip(self, p):
        return self.cip_map.get(p.replace('\\/', '/').replace(',', ''))

    def convert(self, **kwargs):
        #predicts = list(map(lambda iid, l1, l2: {'iid':iid, 'l1':l1, 'l2':l2}, \
        #                df['iid'].values, df['l1'].values, df['l2'].values))
        df = kwargs.get('predict')
        for gid, sty, predict in zip(
            kwargs.get('global_id'),\
            kwargs.get('slice_type'),\
            df['l2'].values):
            cip = self.get_cip(predict)
            if cip:
                yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'cip': cip,
                }

    def send(self, **kwargs):
        gid = kwargs.get('global_id')
        cip = kw['cip']
        table = JournalArticle.rebuild(global_id=gid)
        if table.techdomain:
            table.techdomain[0]['techdomian_id'] = str(cip)
        else:
            table.techdomain = [{"techdomian_id":str(cip),}]
        table.save()


class Org(PC):
    """Prediction consumer for organization"""
    def __init__(self, **kwargs):
        super(Org, self).__init__(**kwargs)

    def convert(self, **kwargs):
        """Internal conversion"""
        for gid, sty, predict in zip(
            kwargs.get('global_id'),\
            kwargs.get('slice_type'),\
            kwargs.get('predict')):
            kw = {
                'global_id': gid,
                'slice_type': sty,
                }
            kw['data'] = ''# TODO add predict 
            yield kw


def start():
    hdr = Article()
    for r in hdr.run_async():
        print(r['future'].get(timeout=5))
     

if __name__ == '__main__':
    start()
