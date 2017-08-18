from read_aps import read_header, read_data, load_labels, get_label, init_plot, plt
import matplotlib.gridspec as gridspec
from os.path import isfile, basename
import argparse
import numpy as np
import glob

class Supervisor(object):
  def __init__(self, name=None):
    self.start = 0
    self.name = name

  def __enter__(self):
    self.start = datetime.now()
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    stack_fmt = ('/{}'*len(inspect.stack())).format(*[s[3] for s in inspect.stack()])
    print("[{}] {}| Elapsed:{}".format(stack_fmt, self.name, datetime.now()-self.start))

modes = ['train', 'test', 'eval']

def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=modes)
  parser.add_argument('--model_id', default='model-{}'.format(np.random.randint(0xffff)), type=str, help='Prefix for model persistance')
  parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
  parser.add_argument('--init_step', default=0, type=int, help='Initial global step')
  parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
  parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
  parser.add_argument('--momentum', default=9e-2, type=float, help='Momentum value')
  parser.add_argument('--decay_rate', default=1e-3, type=float, help='Decay rate')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
  parser.add_argument('--data_root', default='.', type=str, help='Data root')
  parser.add_argument('--label_path', default='.', type=str, help='Label path')
  parser.add_argument('--model_root', default='.', type=str, help='Model path root')
  parser.add_argument('--chkpt', default='.', type=str, help='Check point path')
  parser.add_argument('--outpath', default='out', type=str, help='Path to output')
  args = parser.parse_args()
  return args

def init_axs(tot, rows, fig_img):
  axs = []
  gs = gridspec.GridSpec(rows, tot//rows)
  for i in range(tot):
    axs.append(fig_img.add_subplot(gs[i]))
    axs[-1].set_facecolor('black')
    axs[-1].autoscale(True)
  return axs

def plot_img(x, axs, fig_img):
  data = x.data.numpy()
  data = np.squeeze(data)
  for i in range(data.shape[0]):
    axs[i%len(axs)].imshow(data[i,:,:])
    fig_img.canvas.draw()

def reinit_plot():
  init_plot()
  fig_img = plt.figure(figsize=(8, 8), edgecolor='black', facecolor='black')
  fig_img.suptitle('X')
  fig_loss = plt.figure(figsize=(4, 4), edgecolor='black', facecolor='black')
  fig_loss.suptitle('loss')
  ax = fig_loss.add_subplot(111)
  ax.set_facecolor('black')
  ax.autoscale(True)
  return fig_loss, ax, fig_img

def data_generator(data_root, label_path):
  labels = load_labels(label_path)
  shuffle_grp = glob.glob(data_root+'/*')
  np.random.shuffle(shuffle_grp)

  for src in shuffle_grp:#glob.iglob(data_root+'/*'):
    header = read_header(src)
    data, _ = read_data(src, header)
    iid = basename(src).split('.')[0]
    #data = data.reshape(data.shape[2], 512, 660)
    ys = []
    for i in np.random.permutation(data.shape[2]):
      y = get_label(labels, iid, i)
      #y = y[0] if y else 0
      #yield data[:,:,i], np.array([y])
      ys.append([1] if y else [0])
    yield data.reshape(data.shape[2], 512*660), np.array(ys)


