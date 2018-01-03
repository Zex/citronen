# Feed dict producer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
from boto3.dynamodb.conditions import Key, Attr
from julian.input.producer import FeedDict as FDProducer
from julian.common.topic import Topic
from julian.common.config import get_config
from julian.common.shared import Default
from src.dynamodb.core.broker import get_brk
from src.dynamodb.tables.journal_article import JournalArticle
import logging


class Article(FDProducer):

    def __init__(self, **kwargs):
        super(Article, self).__init__(topic=Topic.INPUT_TECH, **kwargs)
        self.total = 0

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        def extract_x(obj):
            if not obj.get('abstract'):
                return
            iid.append(obj.get('npl_id'))
            in_x.append(obj.get('abstract'))

        config = get_config()
        chunksize = int(getattr(config, 'producer_chunksize', Default.PRODUCER_CHUNKSIZE))

        brk = get_brk(JournalArticle.ID)
        gen = brk._iscan(chunksize=chunksize, \
                filter_expr=Attr('data_type').eq(JournalArticle.DT.value))

        for chunk in gen:
            if not chunk:
                continue
            chunk = chunk.get('Items')
            iid, in_x = [], []

            list(map(extract_x, chunk))
            if iid and in_x:
                self.total += len(in_x)
                print(self, self.total)
                yield {
                    'id': iid,
                    'input_x': in_x,
                    }
