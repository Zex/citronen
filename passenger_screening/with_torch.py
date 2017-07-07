from read_aps import read_header, read_data, load_labels, get_label, init_plot
import torch
from torch import from_numpy
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import CrossEntropyLoss
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import ReLU
from torch.autograd import Variable
from torch.optim import SGD
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
from pandas import read_csv, DataFrame
from os.path import isfile, basename
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
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--momentum', default=0.09, type=float, help='Momentum value')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='Weight decay rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--data_root', default='.', type=str, help='Data root')
    parser.add_argument('--label_path', default='.', type=str, help='Label path')
    parser.add_argument('--model_root', default='.', type=str, help='Model path root')
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
      Conv2d(256, 64,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      ReLU(False),
      MaxPool2d(kernel_size=3)
      )
    # classification
    self.classifier = Sequential(
      Linear(1*64*5*7, 128),
      ReLU(False),
      Dropout(0.3, False),
      Linear(128, 64),
      ReLU(False),
      Dropout(0.3, False),
      Linear(64, 2),
      )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

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


def data_generator(data_root, label_path):
  labels = load_labels(label_path)
  for src in glob.glob(data_root+'/*.aps'):
    header = read_header(src)
    data, _ = read_data(src, header)
    iid = basename(src).split('.')[0]
    print(src, iid, data.shape)
    data = data.reshape(16, 512, 660)
    #y = []
    for i in range(data.shape[0]):
      #y.extend(labels[labels['Id'] == lid]['Probability'].values)
      y = get_label(labels, iid, i)
      yield data[i], y

def start():
  args = PassengerScreening.init()
  chkpt_path = '{}/{}.chkpt'.format(args.model_root, args.model_id)
  # setup
  model = PassengerScreening()
  loss_fn = CrossEntropyLoss().cpu()
  optimizer = SGD(model.parameters(), args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

  # begin
  data_root = args.data_root
  losses = []
  try:
    for i, (data, y) in enumerate(data_generator(data_root, args.label_path)):
      with Supervisor():
        print('data.shape', data.shape, 'y.shape', y.shape)
        if y.shape[0] == 0:
          continue
        data = data.astype(np.float32)
        data = torch.from_numpy(data).contiguous().view(1, 1, 512, 660)
        X = Variable(data, volatile=False)

        #y = np.tile(y, (1, 1, 1, 1)).transpose()
        y = y.astype(np.int64)
        y = torch.from_numpy(y)
        y = Variable(y, volatile=False)

        loss = step(model, optimizer, loss_fn, X, y)
        if loss:
          losses.append(loss)
      torch.save({
          'epoch': i+1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, chkpt_path)
  except Exception as ex:
    print('epoch failed:', ex)
    raise

def plot_loss(losses):
  global fig, ax
  ax.plot(np.arange(len(losses)), losses, '.', color='blue', markerfacecolor='blue')
  fig.canvas.draw()

def step(model, optimizer, loss_fn, data, label):
  loss = None
  try:
    output = model(data)
    #output = output.squeeze()
    print('output: {}: {}'.format(output.dim(), output))
    pred = F.softmax(output)
    print('pred: {}: {}'.format(pred.dim(), pred))
    optimizer.zero_grad()
    print('label: {}: {}'.format(label.dim(), label))
    loss = loss_fn(pred, label)
    print('loss: {}'.format(loss.data))
    loss.backward() 
    optimizer.step()
  except Exception as ex:
    print('step failed:', ex)
    raise
  return loss


if __name__ == '__main__':
  global fig, ax
  init_plot()
  fig = plt.figure(figsize=(16, 16), edgecolor='black', facecolor='black')
  ax = fig.add_subplot(111)
  start()

