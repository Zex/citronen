# Input producer
# Author: Zex Li <top_zlynch@yahoo.com>
from kafka import KafkaProducer
import msgpack
from julian.common.config import get_config
from julian.input.base import Input
from src.dynamodb.tables.utils import get_table


class FeedDict(Input):
    """Feed dict producer"""
    def __init__(self, **kwargs):
        super(FeedDict, self).__init__(**kwargs)
        brokers = get_config().brokers
        self.topic = kwargs.get('topic')
        client_id = kwargs.get('client_id')

        kw = {
                'bootstrap_servers': brokers,
                'value_serializer': msgpack.dumps,
            }

        if self.client_id:
            kw['client_id'] = client_id

        self.pro = KafkaProducer(**kw)
        self.data_type = kwargs.get('data_type')
        self.table = get_table(self.data_type)

    def __del__(self):
        if hasattr(self, 'pro') and self.pro:
            self.pro.flush()
            self.pro.close()

    def fetch(self, **kwargs):
        """Fetch resource from database"""
        yield kwargs

    def send(self, **kwargs):
        """Send to broker
        Args:

        global_id: Global unique ID
        slice_type: Slice type for record
        input_x: Batch of raw text for data provider
        """
        return {'future':self.pro.send(self.topic, msgpack.dumps(kwargs))}

    def convert(self, **kwargs):
        """Internal conversion"""
        return kwargs
