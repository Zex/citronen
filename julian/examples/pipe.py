# P test
from julian.common.pipe import Pipe


class P(Pipe):

    def __init__(self):
        super(P, self).__init__()

    def fetch(self, **kwargs):
        print('fetch', kwargs)
        for i in range(kwargs.get('total')):
            yield {'input_x': i}

    def send(self, **kwargs):
        print('send', kwargs)
        return kwargs

    def convert(self, **kwargs):
        print('convert', kwargs)
        kwargs['input_x'] *= 10
        return kwargs

    def run_async(self, **kwargs):
        for x in self.fetch(**kwargs):
            ret = self.convert(**x)
            yield self.send(**ret)
            break
        for x in self.fetch(**kwargs):
            ret = self.convert(**x)
            yield self.send(**ret)


def selftest():
    p = P()
    res = p.run_async(total=10)
    for r in res:
        print('future', r)


if __name__ == '__main__':
    selftest()
