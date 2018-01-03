# Plot evaluation result
# Author: Zex Li <top_zlynch@yahoo.com>
from datetime import datetime
import ujson
import os


def gen_result(path):
    with open(path) as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            raw = ujson.loads(line)
            yield raw


def extract_result(raw):
    ts = list(raw.keys())
    loss = list(map(lambda e: e['train_err'], raw.values()))
    return ts,loss


def start():
    path = "models/iceberg/logs/eval.json"
    res = list(map(extract_result, gen_result(path)))
    ts, loss = [], []

    for t, l in res:
        ts.extend(t)
        loss.extend(l)
    
    plt.plot(ts, loss)
    plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    start()
