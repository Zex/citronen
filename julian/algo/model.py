# Model handler shotcut
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
import logging
import boto3
from tensorflow.contrib import learn
import tensorflow as tf
from julian.algo.config import get_config


class Inference(object):

    def __init__(self):
        super(Inference, self).__init__()

    def restore(self, sess):
        logging.info('++ [graph] {}'.format(self.graph_path))
        self.saver = tf.train.import_meta_graph(self.graph_path)
        self.saver.restore(sess, self.model_path.split('.')[0])
        graph = tf.get_default_graph()
        self.input_x = graph.get_tensor_by_name("input_x:0")
        self.dropout_keep = graph.get_tensor_by_name("dropout_keep:0")
        self.pred = graph.get_tensor_by_name("pred:0")

    def setup_s3cli(self):
        cfg = get_config()
        self.bucket_name = cfg.s3['bucket']
        self.s3_cli = boto3.resource('s3').meta.client

    def setup_model(self):
        cfg = get_config()
        cfg = getattr(cfg, self.__class__.__name__.lower())

        self.graph_path = os.path.join(cfg['model_base'], cfg['model']['graph'])
        self.model_path = os.path.join(cfg['model_base'], cfg['model']['path'])
        self.vocab_path = os.path.join(cfg['data_base'], cfg['data']['vocab_path'])

        list(map(lambda p:self.fetch_from_s3(\
                os.path.join(cfg['remote_model_base'], p),
                os.path.join(cfg['model_base'], p), cfg.get('force_fetch')),
                cfg['model'].values()))

        list(map(lambda p:self.fetch_from_s3(\
                os.path.join(cfg['remote_data_base'], p),
                os.path.join(cfg['data_base'], p), cfg.get('force_fetch')),
                cfg['data'].values()))

        self.vocab_processor = \
                learn.preprocessing.VocabularyProcessor.restore(self.vocab_path)
        self.sess = tf.Session()
        self.restore(self.sess)

    def infer(self, in_x):
        if not in_x:
            return None
        x = list(self.vocab_processor.transform(in_x))
        feed_dict = {
            self.input_x: x,
            self.dropout_keep: 1.0,
        }
        pred = self.sess.run([self.pred], feed_dict=feed_dict)
        return self.decode(pred)

    def fetch_from_s3(self, src, dest, force=False):
        ddir = os.path.dirname(dest)

        if not os.path.isdir(ddir):
            os.makedirs(ddir)

        if os.path.isfile(dest) and not force:
            return

        try:
            logging.info("Fetching {}/{} to {}".format(self.bucket_name, src, dest))
            self.s3_cli.download_file(self.bucket_name, src, dest)
            if os.path.isfile(dest):
                logging.info("{} downloaded".format(dest))
        except Exception as ex:
            logging.error("Exception on fetching model: {}".format(ex))

if __name__ == '__main__':
    start()
