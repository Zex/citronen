# Feed dict producer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
from julian.input.producer import FeedDict as FDProducer
from julian.common.topic import Topic
from src.dynamodb.common.utils import DataType, SliceType
from src.dynamodb.tables.journal_article import JournalArticle
from src.dynamodb.tables.news import NewsTable


class Article(FDProducer):

    def __init__(self, **kwargs):
        super(Article, self).__init__(topic=Topic.INPUT_TECH, **kwargs)
        pass

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        #for items in JournalArticle.iscan(chunksize=10)
        for items in NewsTable.iscan(chunksize=10)
            # list of table object
            if not objs:
                continue
            gid = sty = in_x = []
            for obj in objs:
                gid.append(obj.global_id)
                sty.append(obj.slice_type)
                in_x.append(obj.data.abstract)
            print(gid, in_x)
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
        for objs in self.table._iscan(
            Attr('data_type').eq(DataType.ORG.name.lower())&\
            Attr('slice_type').eq(SliceType.BASIC)):
            # list of table object
            if not objs:
                continue
            gid = sty = in_x = []
            for obj in objs:
                gid.append(obj.global_id)
                sty.append(obj.slice_type)
                in_x.append(obj.data.description)
            yield {
                'global_id': gid,
                'slice_type': sty,
                'input_x': in_x,
                }


def start():
    input_hdrs = [Article()]
    ps = [mp.Process(target=hdr.run_async) \
            for hdr in input_hdrs]
    [p.start() for p in ps]
    #[p.join() for p in ps]
     

if __name__ == '__main__':
    start()
