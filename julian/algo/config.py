# Service configure
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import sys
from os import environ as env
import yaml
import string
import logging

class Config(object):

    def __init__(self):
        super(Config, self).__init__()
        self.load_config()

    def load_config(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.yaml')
        self.load_yaml()

    def load_yaml(self):
        with open(self.path) as fd:
            cfg = yaml.load(fd)
        _ = [setattr(self, k.lower(), v) for k, v in cfg.items()]

    def load_env(self):
        _ = [setattr(self, k.lower().lstrip(string.digits+string.punctuation), v) \
                for k, v in env.items()]

    def raise_on_not_set(self, name):
        if not name:
            return

        if not hasattr(self, name):
            raise AttributeError("{} not set".format(name.upper()))

    def __str__(self):
        return '\n'.join(['{}:{}'.format(k, v) for k, v in self.__dict__.items()])

def get_config():
    if not globals().get('julian_config'):
        globals()['julian_config'] = Config()
    return globals().get('julian_config')


def get_logger():
    if not globals().get('julian_logger'):
        lgr = logging.getLogger()
        lgr.setLevel(logging.INFO)
        lgr.addHandler(logging.StreamHandler(stream=sys.stdout))
        globals()['julian_logger'] = lgr
    return globals()['julian_logger']
