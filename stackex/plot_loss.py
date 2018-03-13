import os
import ujson
import numpy as np
import glob
import seaborn as sns
from matplotlib import pyplot as plt


def plot_loss(loss, kl, recon):
    plt.scatter(np.arange(len(loss)), loss, label='vae')
    plt.plot(np.arange(len(kl)), kl, label='kl')
    plt.plot(np.arange(len(recon)), recon, label='reccon')
    plt.legend()
    plt.show()


def foreach_line(path):
    with open(path) as fd:
        while True:
            line = fd.readline()
            if not line:
                break
            yield line


def foreach_loss(base_dir):
    loss_paths = glob.glob(base_dir+'/samples_*.json')
    loss_paths = sorted(loss_paths, key=lambda p: os.stat(p).st_mtime)

    for path in loss_paths[-1:]:
        loss, kl_loss, recon_loss = [], [], []
        for line in foreach_line(path):
            data = ujson.loads(line)
            loss.append(data.get('loss'))
            kl_loss.append(data.get('kl_loss'))
            recon_loss.append(data.get('recon_loss'))
        plot_loss(loss, kl_loss, recon_loss)

if __name__ == '__main__':
    foreach_loss('data/stackex')
