# Plot evaluation result
# Author: Zex Li <top_zlynch@yahoo.com>
import numpy as np
import pandas as pd
import glob
from datetime import datetime
import ujson
import os


def gen_result(path):
    with open(path) as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            yield ujson.loads(line)


def extract_result(raw):
    ts = list(raw.keys())
    loss = list(map(lambda e: e['train_err'], raw.values()))
    return ts,loss


def load_pred(path):
    df = pd.read_csv(path)
    return df['is_iceberg'].values

def plot_pred():
    colors = ['g', 'b', 'y', 'k', 'l']
    #colors = list(matplotlib.colors.cnames.keys())
    lbls = []
    fig = plt.figure(facecolor='darkred', edgecolor='k')
    ax = fig.add_subplot(111)
    for i, path in enumerate(glob.glob('data/iceberg/pred*csv')):
        vals = load_pred(path)
        ax.scatter(range(len(vals)), vals, color=colors[i], s=3)
        lbls.append(os.path.basename(path).split('.')[0])
    ax.legend(lbls)
    ax.set_facecolor('darkred')
    plt.show()

def plot_eval():
    path = "models/iceberg/logs/eval.json"
    res = list(map(extract_result, gen_result(path)))
    ts, loss = [], []

    for t, l in res:
        ts.extend(t)
        loss.extend(l)
    
    plt.plot(range(len(loss)), loss)
    plt.show()

def start():
    plot_pred()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    start()
