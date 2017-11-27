# Model handler
# Author: Zex Li <top_zlynch@yahoo.com>
import os, glob
from enum import Enum, unique
import pandas as pd
import boto3
from julian.with_tf import Julian, init
from julian.config import *

@unique
class MODE(Enum):
    STREAM = 1
    SINGLE = 2
    COMPAT = 3


class ModelHandler(object):

    def __init__(self):

        if not AWS_ACCESS_KEY:
            raise RuntimeError("AWS_ACCESS_KEY not given")
        if not AWS_SECRET_KEY:
            raise RuntimeError("AWS_SECRET_KEY not given")
        if not AWS_REGION:
            raise RuntimeError("AWS_REGION not given")
        if not AWS_S3_BUCKET:
            raise RuntimeError("AWS_S3_BUCKET not given")

        self.session = boto3.session.Session(
          aws_access_key_id=AWS_ACCESS_KEY,
          aws_secret_access_key=AWS_SECRET_KEY,
          region_name=AWS_REGION
        )
        self.s3 = self.session.client('s3')
        self.julian = None

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def fetch_from_s3(self, src, dest, force=False):
        ddir = os.path.dirname(dest)
        if not os.path.isdir(ddir):
            os.makedirs(ddir)

        if not glob.glob(ddir) and not force:
            return

        try:
            print("Fetching {} to {}".format(src, dest))
            self.s3.download_file(AWS_S3_BUCKET, src, dest)
        except Exception as ex:
            print("Exception on fetching model: {}".format(ex))

    def init_queues(self):
        sqs = boto3.resource('sqs')
        self.in_queue = sqs.get_queue_by_name(QueueName='AWS_SQS_INPUT')
        self.out_queue = sqs.get_queue_by_name(QueueName='AWS_SQS_OUTPUT')

    def predict(self, x):
        if not self.julian:
            return
        feed_dict = {
            self.julian.input_x: x,
            self.julian.dropout_keep: self.julian.dropout,
        }
        pred = sess.run([self.julian.pred], feed_dict=feed_dict)
        return self.provider.decode(pred)
