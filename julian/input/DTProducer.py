# Feed dict producer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
import multiprocessing as mp
import pickle, ujson, glob, os #TODO REMOVE LATER
from julian.input.producer import FeedDict as FDProducer
from julian.common.topic import Topic
from julian.common.config import get_config
from julian.common.shared import Default
from src.dynamodb.common.shared import DataType, SliceType
from src.dynamodb.core.global_table import Global
from src.dynamodb.tables.journal_article import JournalArticle
from src.dynamodb.tables.organization import OrganizationTable


class Article(FDProducer):

    def __init__(self, **kwargs):
        super(Article, self).__init__(topic=Topic.INPUT_TECH, **kwargs)
        pass

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        def extract_x(obj):
            # TODO REMOVE LATER
            obj = Global(ujson.loads(obj))
            obj.data = JournalArticle(ujson.loads(obj.data))
            if not getattr(obj.data, 'abstract', None):
                return

            gid.append(obj.global_id)
            sty.append(obj.slice_type)
            in_x.append(obj.data.abstract)

        config = get_config()
        chunksize = int(getattr(config, 'producer_chunksize', Default.PRODUCER_CHUNKSIZE))

        #for objs in JournalArticle.iquery(chunksize=chunksize, verbose=True):
        #    if not objs:
        #        continue
        #TODO REMOVE LATER
        with open('julian/tools/ja-100', 'rb') as fd:
            objs = pickle.load(fd)
            gid, sty, in_x = [], [], []
            list(map(extract_x, objs))
            if gid and sty and in_x:
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
            # TODO REMOVE LATER
            obj = Global(ujson.loads(obj))
            obj.data = OrganizationTable(ujson.loads(obj.data))
            if not getattr(obj.data, 'description', None):
                return
            gid.append(obj.global_id)
            sty.append(obj.slice_type)
            in_x.append(obj.data.description)

        config = get_config()
        chunksize = int(getattr(config, 'producer_chunksize', Default.PRODUCER_CHUNKSIZE))

        #for objs in OrganizationTable.iquery(chunksize=chunksize, verbose=True):
        #    if not objs:
        #        continue
        #TODO REMOVE LATER
        for f in glob.iglob('julian/tools/org.pickle.enum/*'):
            with open(f, 'rb') as fd:
                objs = pickle.load(fd)

            gid, sty, in_x = [], [], []
            list(map(extract_x, objs))
            if gid and sty and in_x:
                yield {
                    'global_id': gid,
                    'slice_type': sty,
                    'input_x': in_x,
                    }


def run_async(hdr_name):
    try:
        hdr = globals().get(hdr_name)()
        _ = list(hdr.run_async())
    except KeyboardInterrupt:
        print("++ [terminate] {}".format(hdr_name))


def start():
    pool = []
    try:
        for hdr_name in get_config().producers.split(','):
            pool.append(mp.Process(target=run_async,
                                    args=(hdr_name,),
                                    name=hdr_name))
        list(map(lambda p: p.start(), pool))
    except KeyboardInterrupt:
        #list(map(lambda p: p.join(), pool))
        print("++ [terminate]")
     

if __name__ == '__main__':
    start()
