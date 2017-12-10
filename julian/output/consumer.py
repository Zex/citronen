# Output producer
# Author: Zex Li <top_zlynch@yahoo.com>
from kafka import KafkaConsumer
import msgpack
from julian.common.config import get_config
from julian.input.base import Output
from julian.common.topic import Topic
from src.dynamodb.tables.utils import get_table


class Prediction(Output):
    """Prediction Consumer"""
    def __init__(self, *args, **kwargs):
        super(Prediction, self).__init__()
        brokers = get_config().brokers
        client_id = kwargs.get('client_id')

        kw = {
                'bootstrap_servers': brokers,
                'value_deserializer': msgpack.unpackb,
            }

        if self.client_id:
            kw['client_id'] = client_id

        self.con = KafkaConsumer(Topic.PREDICT,**kw)
        self.data_type = kwargs.get('data_type')
        self.table = get_table(self.data_type)

    def __del__(self):
        if hasattr(self, 'con') and self.con:
            self.con.close()

    def fetch(self, **kwargs):
        """Fetch results from model"""
        for msg in self.con:
            yield {msg.key:msg.value}

    def send(self, **kwargs):
        """Write back to database
        """
        return self.table.update(kwargs)

    def convert(self, **kwargs):
        """Internal conversion"""
        return kwargs
