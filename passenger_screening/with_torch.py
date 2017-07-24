from common import init, plot_img, init_axs, data_generator, reinit_plot
from datetime import datetime
from pandas import read_csv, DataFrame
from os.path import isfile, basename
from sys import stderr
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import inspect
import torch
from torch import from_numpy
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import AvgPool2d
from torch.nn import CrossEntropyLoss
from torch.nn import BatchNorm2d
from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import ReLU
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim import RMSprop
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid


class PassengerScreening(Module):

  def __init__(self):
    super(PassengerScreening, self).__init__()
    self.total_class = 2
    self.features = Sequential(
      Conv2d(1, 64,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(64),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(64, 128,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(128),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(128, 256,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(256),
      ReLU(False),
      MaxPool2d(kernel_size=3),
      Conv2d(256, 220,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(220),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      Conv2d(220, 100,
             kernel_size=3,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(100),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      Conv2d(100, 64,
             kernel_size=2,
             stride=1,
             padding=2,
             bias=True),
      BatchNorm2d(64),
      ReLU(False),
      MaxPool2d(kernel_size=2),
      Conv2d(64, 1,
             kernel_size=2,
             stride=1,
             padding=0,
             bias=True),
      BatchNorm2d(1),
      ReLU(False),
      AvgPool2d(kernel_size=2),
      )
    # classification
    self.classifier = Sequential(
      Linear(1*64*2*3, self.total_class),
      ReLU(False),
      Dropout(0.5, False),
      )

#    self.axs = init_axs(64, 8)

  def forward(self, x):
    x = self.features(x)
    print('features', x.data.numpy().shape)
    # plot im data
    #plot_img(x, self.axs)
    x = x.view(x.size(0), -1)
    print('view', x.data.numpy().shape)
    #x = self.classifier(x)
    #print('classification', x.data.numpy().shape)
    return x

def accuracy(output, target, topk=5):
  ret, pred = output.topk(topk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  corr_k = correct[:topk].view(-1).float().sum(0)
  return correct, corr_k

def start():
  args = init()
  model_path = '{}/{}.chkpt'.format(args.model_root, args.model_id)
  chkpt_path = args.chkpt
  # setup
  if isfile(chkpt_path):
    print('loading chkpt_path:', chkpt_path)
    model = torch.load(chkpt_path)
    optimizer = model.get('optimizer')
    init_epoch = model.get('epoch')
    model = model.get('model')
    print(optimizer, init_epoch, model)
  else:
    model = PassengerScreening()
    init_epoch = args.init_epoch

  #loss_fn = BCELoss().cpu()
  #loss_fn = CrossEntropyLoss().cpu()
  loss_fn = BCELoss()
  optimizer = RMSprop(model.parameters(), args.lr,
        momentum=args.momentum, 
        weight_decay=args.decay_rate)
  #nor = transforms.Normalize(mean=[0.485, 0.465, 0.406],
  #                          std=[0.229, 0.224, 0.225])
  nor = transforms.Normalize(mean=[0.254, 0.328, 0.671],
                            std=[0.229, 0.224, 0.225])

  losses, accs = [], []
  # begin
  if args.mode == 'train':
    for i in range(args.epochs): 
      global_epoch = init_epoch+i+1
      loss, acc = epoch(model, optimizer, nor, loss_fn, global_epoch, args)
      if not loss or not acc:
        break
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
  elif args.mode == 'test':
    global_epoch = init_epoch
    epoch(model, optimizer, nor, loss_fn, global_epoch, args)
  else: # eval
    pass
    
  return losses, accs

def epoch(model, optimizer, nor, loss_fn, global_epoch, args):
  def get_x(data):
      data = data.astype(np.float32)
      data = torch.from_numpy(data).contiguous().view(1, 1, 512, 660)
      X = nor(data)
      X = Variable(X, volatile=False)
      return X

  def get_y(label):
      y = label.astype(np.int64)
      y = torch.from_numpy(y)
      y = Variable(y, volatile=False)
      return y

  try:
    loss, acc = None, None
    for i, (data, y) in enumerate(data_generator(args.data_root, args.label_path)):
      if args.mode == 'test':
        output = model(get_x(data))
        print('output', output.data.numpy())
        pred = F.softmax(output)
        print('pred', np.squeeze(pred.data.numpy()), 'label', y)
        continue
         
      if y.shape[0] == 0:
        continue

      loss, acc = step(model, optimizer, loss_fn, get_x(data), get_y(y))
      if not loss or not acc:
        break
      loss = loss.squeeze().data[0]
      acc = acc.squeeze().data[0]
      print('[{}] loss: {:.4f}, acc: {}'.format(global_epoch, loss, acc), flush=True)
  except Exception as ex:
    print('epoch failed:', ex)
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
#    pred = F.softmax(output)
    optimizer.zero_grad()
    pred, acc = accuracy(output, label, topk=1)
    optimizer.zero_grad()
    #loss = loss_fn(output[0][label.data.numpy()[0]], label.float())
    loss = loss_fn(output, label.float())
    print('label', label.float().data.numpy().squeeze(), 
        'output', output.data.numpy().squeeze(),
        'pred', pred.data.numpy().squeeze(),
        'acc', acc.data.numpy().squeeze(),
        'loss', loss.data.numpy().squeeze())
    loss.backward() 
    optimizer.step()
    iter_params(model)
  except Exception as ex:
    print('step failed:', ex)
  return loss, acc

def iter_params(model):
  print('='*20, 'parameters', '='*20, file=stderr)
  for k in model.parameters():
    print(k, file=stderr)


if __name__ == '__main__':
  global fig_img
  #_, _, fig_img = reinit_plot()
  start()

