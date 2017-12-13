# Feed dict producer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
import multiprocessing as mp
import pickle, ujson
from julian.input.producer import FeedDict as FDProducer
from julian.common.topic import Topic
from src.dynamodb.common.shared import DataType, SliceType
from src.dynamodb.tables.journal_article import JournalArticle
from src.dynamodb.core.global_table import Global


class Article(FDProducer):

    def __init__(self, **kwargs):
        super(Article, self).__init__(topic=Topic.INPUT_TECH, **kwargs)
        pass

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        def extract_x(obj):
            obj = Global(ujson.loads(obj))
            obj.data = JournalArticle(ujson.loads(obj.data))
            gid.append(obj.global_id)
            sty.append(obj.slice_type)
            in_x.append(obj.data.abstract if obj.data.abstract else '')
        #for objs in JournalArticle.iscan(chunksize=10, verbose=True):
        #TODO REMOVE LATER
        with open('julian/tools/ja-100', 'rb') as fd:
            objs = pickle.load(fd)
        if True:
            # END REMOVE LATER
            #if not objs:
            #    continue
            gid, sty, in_x = [], [], []
            list(map(extract_x, objs))
            yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'input_x': in_x,
                    }


class Org(FDProducer):

    def __init__(self, **kwargs):
        super(Org, self).__init__(topic=Topic.INPUT_NAICS, **kwargs)
        pass

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        def extract_x(obj):
            gid.append(obj.global_id)
            sty.append(obj.slice_type)
            in_x.append(obj.data.description)
            return gid, sty, in_x

        for objs in self.table._iscan(
            Attr('data_type').eq(DataType.ORG.name.lower())&\
            Attr('slice_type').eq(SliceType.BASIC)):
            # list of table object
            if not objs:
                continue
            gid, sty, in_x = [], [], []
            list(map(extract_x, objs))
            yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'input_x': in_x,
                    }


def start():
    hdr = Article()
    for r in hdr.run_async():
        print(r['future'].get(timeout=5))
     

if __name__ == '__main__':
    start()
