# Feed dict producer for each data type
# Author: Zex Li <top_zlynch@yahoo.com>
from boto3.dynamodb.conditions import Key, Attr
from boto3.dynamodb.types import Binary
from julian.input.producer import FeedDict as FDProducer
from julian.common.topic import Topic
from julian.common.config import get_config
from julian.common.shared import Default
from src.dynamodb.common.utils import json_unzip
from src.dynamodb.core.broker import get_brk
from src.dynamodb.tables.organization import OrganizationTable
import gzip
import logging


class Org(FDProducer):

    def __init__(self, **kwargs):
        super(Org, self).__init__(topic=Topic.INPUT_NAICS, **kwargs)
        self.total = 0
        pass

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        def extract_x(obj):
            if not obj.get('description'):
                return
            val = obj.get('description')
            if val and isinstance(val, Binary):
                val = gzip.decompress(val.value)
            in_x.append(val)
            iid.append(obj.get('entity_id'))

        config = get_config()
        chunksize = int(getattr(config, 'producer_chunksize', Default.PRODUCER_CHUNKSIZE))

        brk = get_brk(OrganizationTable.ID)
        gen = brk._iscan(chunksize=chunksize, \
                filter_expr=Attr('data_type').eq(OrganizationTable.DT.value))

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
