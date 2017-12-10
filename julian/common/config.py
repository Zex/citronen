# Service configure
# Author: Zex Li <top_zlynch@yahoo.com>
from os import environ as env

class Config(object):

    def __init__(self):
        super(Config, self).__init__()
        self.load_config()

    def load_config(self):
        self.load_env()

    def load_env(self):
        [setattr(self, k.lower(), v) for k, v in env.items()]

    def raise_on_not_set(self, name):
        if not name:
            return

        if not hasattr(self, name):
            raise AttributeError("{} not set".format(name.upper()))


def get_config():
    if not globals().get('julian_config'):
        globals()['julian_config'] = Config()
    return globals().get('julian_config')
