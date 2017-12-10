# Prediction consumer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
from julian.input.consumer import Prediction as PC
from src.dynamodb.common.shared import DataType
import multiprocessing as mp


class Article(PC):
    """Prediction consumer for article"""
    def __init__(self, **kwargs):
        kwargs['data_type'] = DataType.J_ARTICLE
        super(Article, self).__init__(**kwargs)

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

class Org(PC):
    """Prediction consumer for organization"""
    def __init__(self, **kwargs):
        kwargs['data_type'] = DataType.ORG
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
    ps = []
    output_hdrs = [Article(), Org()]
    [ps.append(mp.Process(target=hdr.run_async)) \
            for hdr in output_hdrs]
    [p.start() for p in ps]
    [p.join() for p in ps]
     

if __name__ == '__main__':
    start()
