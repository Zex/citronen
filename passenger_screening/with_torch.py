from read_aps import read_header, read_data, load_labels, get_label, init_plot, plt
import torch
from torch import from_numpy
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import CrossEntropyLoss
from torch.nn import BCELoss
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import ReLU
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim import RMSprop
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from datetime import datetime
from pandas import read_csv, DataFrame
from os.path import isfile, basename
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import inspect
import glob


class PassengerScreening(Module):

  modes = ['train', 'test', 'eval']

  @staticmethod
  def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=PassengerScreening.modes)
    parser.add_argument('--model_id', default='model-{}'.format(np.random.randint(0xffff)), type=str, help='Prefix for model persistance')
    parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
    parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
    parser.add_argument('--lr', default=1e-2, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=9e-2, type=float, help='Momentum value')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--data_root', default='.', type=str, help='Data root')
    parser.add_argument('--label_path', default='.', type=str, help='Label path')
    parser.add_argument('--model_root', default='.', type=str, help='Model path root')
    parser.add_argument('--chkpt', default='.', type=str, help='Check point path')
    args = parser.parse_args()
    return args

  def __init__(self):
    super(PassengerScreening, self).__init__()
    self.total_class = 2
    self.features = Sequential(
      Conv2d(1, 64,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(64, 128,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(128, 256,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(256, 220,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      Conv2d(220, 100,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      Conv2d(100, 64,
             kernel_size=3,
             stride=1,
             padding=2,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      )
    # classification
    self.classifier = Sequential(
      Linear(1*64*2*3, 64),
      ReLU(False),
      Dropout(0.3, False),
      Linear(64, 32),
      ReLU(False),
      Dropout(0.3, False),
      Linear(32, self.total_class),
      )

#    self.axs = init_axs(64, 8)

  def forward(self, x):
    x = self.features(x)
    print('features', x.data.numpy().shape)
    # plot im data
#    plot_img(x, self.axs)
    x = x.view(x.size(0), -1)
    print('view', x.data.numpy().shape)
    x = self.classifier(x)
    print('classification', x.data.numpy().shape)
    return x


def init_axs(tot, rows):
  axs = []
  gs = gridspec.GridSpec(rows, tot//rows)
  for i in range(tot):
    axs.append(fig_img.add_subplot(gs[i]))
    axs[-1].set_facecolor('black')
    axs[-1].autoscale(True)
  return axs

def plot_img(x, axs):
  data = x.data.numpy()
  data = np.squeeze(data)
  for i in range(data.shape[0]):
    axs[i%len(axs)].imshow(data[i,:,:])
    fig_img.canvas.draw()

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

def accuracy(output, target, topk=5):
  ret, pred = output.topk(topk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  corr_k = correct[:topk].view(-1).float().sum(0)
  return correct, corr_k

def data_generator(data_root, label_path):
  labels = load_labels(label_path)
  for src in glob.glob(data_root+'/*.aps'):
    header = read_header(src)
    data, _ = read_data(src, header)
    iid = basename(src).split('.')[0]
#    print(src, iid, data.shape)
    data = data.reshape(16, 512, 660)
    #y = []
    for i in range(data.shape[0]):
      y = get_label(labels, iid, i)
      yield data[i], y

def start():
  args = PassengerScreening.init()
  model_path = '{}/{}.chkpt'.format(args.model_root, args.model_id)
  chkpt_path = args.chkpt
  # setup
  if isfile(chkpt_path):
    print('loading chkpt_path:', chkpt_path)
    model = torch.load(chkpt_path)
    optimizer = model.get('optimizer')
    init_epoch = model.get('epoch')
    model = model.get('model')
    #print(type(optimizer), type(init_epoch), type(model))
  else:
    model = PassengerScreening()
    init_epoch = args.init_epoch

  #loss_fn = BCELoss().cpu()
  loss_fn = CrossEntropyLoss().cpu()
#  optimizer = SGD(model.parameters(), args.lr, 
#        momentum=args.momentum, 
#        weight_decay=args.weight_decay)
  optimizer = RMSprop(model.parameters(), args.lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
  nor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

  # begin
  losses, accs = [], []
  for i in range(args.epochs): 
    global_epoch = init_epoch+i+1
    loss, acc = epoch(model, optimizer, nor, loss_fn, global_epoch, args)

    if loss:
      losses.append(loss)
    if acc:
      accs.append(acc)
    print('[{}] loss: {:.4f}, acc: {}, losses nr: {}'.format(global_epoch, loss, acc, len(losses)), flush=True)
    torch.save({
        'epoch': global_epoch,
        'model': model,
        'optimizer': optimizer,
        }, '{}-{}-{:.4f}'.format(model_path, global_epoch, loss))
  return losses, accs

def epoch(model, optimizer, nor, loss_fn, global_epoch, args):
  try:
    loss, acc = None, None
    for i, (data, y) in enumerate(data_generator(args.data_root, args.label_path)):
      if y.shape[0] == 0:
        continue
      data = data.astype(np.float32)
      data = torch.from_numpy(data).contiguous().view(1, 1, 512, 660)
      X = nor(data)
      X = Variable(X, volatile=False)

      y = y.astype(np.int64)
      y = torch.from_numpy(y)
      y = Variable(y, volatile=False)

      loss, acc = step(model, optimizer, loss_fn, X, y)
      loss = loss.squeeze().data[0]
      acc = acc.squeeze().data[0]
      print('[{}] loss: {:.4f}, acc: {}'.format(global_epoch, loss, acc, flush=True))
  except Exception as ex:
    print('epoch failed:', ex)
    raise
  return loss, acc

def plot_loss(losses):
  global fig_loss, ax
  ax.plot(np.arange(len(losses)), losses, '.', color='blue', markerfacecolor='blue')
  fig_loss.canvas.draw()

def step(model, optimizer, loss_fn, data, label):
  loss = None
  try:
    output = model(data)
#    output = output.view(-1, 2)
#    output = output.squeeze()
#    print('output', output)
#    pred = F.softmax(output)
#    print('pred', pred.data[0][0], pred.data[0][1], 'label', label.data[0])
    optimizer.zero_grad()
    pred, acc = accuracy(output, label, topk=2)
    print('pred', pred.data[0][0], ',', pred.data[1][0], 'target', label.data[0])
    optimizer.zero_grad()
    loss = loss_fn(pred, label)
    loss.backward() 
    optimizer.step()
  except Exception as ex:
    print('step failed:', ex)
    raise
  return loss, acc


if __name__ == '__main__':
  global fig_loss, ax, fig_img
  init_plot()
  fig_img = plt.figure(figsize=(8, 8), edgecolor='black', facecolor='black')
  fig_img.suptitle('X')
  fig_loss = plt.figure(figsize=(4, 4), edgecolor='black', facecolor='black')
  fig_loss.suptitle('loss')
  ax = fig_loss.add_subplot(111)
  ax.set_facecolor('black')
  ax.autoscale(True)
  start()

