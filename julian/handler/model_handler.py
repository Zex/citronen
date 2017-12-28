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
        self.setup_s3cli()
        self.setup_kafka(**kwargs)

    def setup_s3cli(self):
        config = get_config()
        if getattr(config, 'aws_access_key', None) and \
                getattr(config, 'aws_secret_key', None):
            self.s3_cli = boto3.client('s3',
                    aws_access_key=config.aws_access_key,
                    aws_secret_key=config.aws_secret_key,
                    )
        else:
            self.s3_cli = boto3.resource('s3').meta.client

    def setup_kafka(self, **kwargs):
        """
        Setup kafka consumer and producer

        Args:
        topic_pair: A `dict` represents in-topic and out-topic
        """
        kw = {'bootstrap_servers': get_config().kafka_brokers.split(','),}
        self.topic_pair = kwargs.get('topic_pair', {})
        self.cons = KafkaConsumer(
                *self.topic_pair.keys(),
                value_deserializer=msgpack.unpackb,
                **kw)
        self.pro = KafkaProducer(
                value_serializer=msgpack.dumps,
                **kw)

    def _precheck(self):
        config = get_config()
        if not (getattr(config, 'local_model', 'false') and bool(config.local_model)):
            config.raise_on_not_set('aws_s3_bucket')
            self.bucket_name = config.aws_s3_bucket

#        config.raise_on_not_set('aws_access_key')
#        config.raise_on_not_set('aws_secret_key')
#        config.raise_on_not_set('aws_region_key')

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def fetch_from_s3(self, src, dest, force=False):
        ddir = os.path.dirname(dest)
        if not os.path.isdir(ddir):
            os.makedirs(ddir)
        if os.path.isfile(dest) and not force:
            return

        try:
            print("Fetching {}/{} to {}".format(self.bucket_name, src, dest))
            self.s3_cli.download_file(self.bucket_name, src, dest)
            if os.path.isfile(dest):
                print("{} downloaded".format(dest))
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
            data.update({'topic': msg.topic})
            yield data

    def convert(self, **kwargs):
        input_x = kwargs.pop('input_x', '')
        input_x = list(map(lambda x: x.decode(), input_x))
        res = self.predict(input_x)
        kwargs.update({
            'predict': res.values.tolist(),
            })
        return kwargs

    def send(self, **kwargs):
        return {'future':self.pro.send(self.topic_pair.get(kwargs['topic']), msgpack.dumps(kwargs))}


    def __del__(self):
        cons = getattr(self, 'cons', None)
        if cons:
            cons.close()

        pro = getattr(self, 'pro', None)
        if pro:
            pro.close()
