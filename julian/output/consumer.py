# Output producer
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import boto3
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

        self.topics = kwargs.get('topics')
        self.cons = KafkaConsumer(*self.topics, **kw)
        self.setup_s3cli()

    def setup_s3cli(self):
        config = get_config()

        if not (getattr(config, 'local_model', 'false') and bool(config.local_model)):
            config.raise_on_not_set('aws_s3_bucket')
            self.bucket_name = config.aws_s3_bucket

        if getattr(config, 'aws_access_key', None) and \
                getattr(config, 'aws_secret_key', None):
            self.s3_cli = boto3.client('s3',
                    aws_access_key=config.aws_access_key,
                    aws_secret_key=config.aws_secret_key,
                    )
        else:
            self.s3_cli = boto3.resource('s3').meta.client

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

    def __del__(self):
        if hasattr(self, 'con') and self.cons:
            self.cons.close()

    def fetch(self, **kwargs):
        """Fetch results from model"""
        for msg in self.cons:
            data = msgpack.unpackb(msg.value)
            data = {k.decode():v for k, v in data.items()}
            yield data

    def send(self, **kwargs):
        """Write back to database
        """
        return kwargs

    def convert(self, **kwargs):
        """Internal conversion"""
        return kwargs

    def run_async(self, **kwargs):
        """Asynchronous run, requires `fetch` to return generator"""
        for x in self.fetch(**kwargs):
            for intm in self.convert(**x):
                if intm:
                    yield self.send(**intm)

    def __del__(self):
        cons = getattr(self, 'cons', None)
        if cons:
            cons.close()
