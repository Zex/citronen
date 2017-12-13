# Output producer
# Author: Zex Li <top_zlynch@yahoo.com>
from kafka import KafkaConsumer
import msgpack
from julian.common.config import get_config
from julian.output.base import Output
from julian.common.topic import Topic


class Prediction(Output):
    """Prediction Consumer"""
    def __init__(self, *args, **kwargs):
        super(Prediction, self).__init__()
        brokers = get_config().kafka_brokers.split(',')
        client_id = kwargs.get('client_id')

        kw = {
                'bootstrap_servers': brokers,
                'value_deserializer': msgpack.unpackb,
            }

        if client_id:
            kw['client_id'] = client_id

        self.con = KafkaConsumer(Topic.PREDICT,**kw)
        self.table = kwargs.get('table')

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
        return kwargs

    def convert(self, **kwargs):
        """Internal conversion"""
        return kwargs
