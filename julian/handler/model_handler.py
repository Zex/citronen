# Model handler
# Author: Zex Li <top_zlynch@yahoo.com>
import os, glob
from enum import Enum, unique
import pandas as pd
import boto3
import tensorflow as tf
import msgpack
from kafka import KafkaProducer, KafkaConsumer
from julian.core.with_tf import Julian
from julian.common.config import get_config
from julian.common.topic import Topic
from julian.common.pipe import Pipe

@unique
class MODE(Enum):
    STREAM = 1
    SINGLE = 2
    COMPAT = 3


class ModelHandler(Pipe):

    def __init__(self, **kwargs):
        super(ModelHandler, self).__init__(**kwargs)
        self._precheck()
        self.julian = None
        self.s3 = boto3.resource('s3')
        self.setup_kafka(**kwargs)

    def setup_kafka(self, **kwargs):
        kw = {'bootstrap_servers': get_config().kafka_brokers.split(','),}

        self.cons = KafkaConsumer(
                Topic.INPUT_TECH,
                Topic.INPUT_NAICS,
                value_deserializer=msgpack.unpackb,
                **kw)
        self.pro = KafkaProducer(
                value_serializer=msgpack.dumps,
                **kw)

    def _precheck(self):
        config = get_config()
#        config.raise_on_not_set('aws_access_key')
#        config.raise_on_not_set('aws_secret_key')
#        config.raise_on_not_set('aws_region_key')
        if not (hasattr(config, 'local_model') and config.local_model):
            config.raise_on_not_set('aws_s3_bucket')
            self.bucket_name = config.aws_s3_bucket

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def fetch_from_s3(self, src, dest, force=False):
        ddir = os.path.dirname(dest)
        if not os.path.isdir(ddir):
            os.makedirs(ddir)
        if os.path.isfile(dest) and not force:
            return

        try:
            print("Fetching {} to {}".format(src, dest))
            self.s3.meta.client.download_file(self.bucket_name, src, dest)
        except Exception as ex:
            print("Exception on fetching model: {}".format(ex))

    def setup_model(self, args):
        self.julian = Julian(args) 
        self.sess = tf.Session()
        self.julian.setup_model(self.sess)

    def predict(self, in_x):
        if not self.julian or not in_x:
            return None
        x = list(self.julian.provider.vocab_processor.transform(in_x))
        feed_dict = {
            self.julian.input_x: x,
            self.julian.dropout_keep: self.julian.dropout,
        }
        pred = self.sess.run([self.julian.pred], feed_dict=feed_dict)
        return self.julian.provider.decode(pred)

    def fetch(self, **kwargs):
        """Fetch from feed dict producer"""
        for msg in self.cons:
            data = msgpack.unpackb(msg.value)
            data = {k.decode():v for k, v in data.items()}
            yield data

    def convert(self, **kwargs):
        input_x = kwargs.pop('input_x', '')
        input_x = list(map(lambda x: x.decode(), input_x))
        res = self.predict(input_x)
        kwargs.update({'predict':res.values.tolist()})
        return kwargs

    def send(self, **kwargs):
        return {'future':self.pro.send(Topic.PREDICT, msgpack.dumps(kwargs))}
