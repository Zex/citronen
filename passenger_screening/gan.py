#import matplotlib
#matplotlib.use("TkAgg")
from common import init, plot_img, init_axs, data_generator, reinit_plot
import glob
import numpy as np
from itertools import chain
from os import mkdir
from os.path import isdir
import seaborn as sn
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d
from torch.nn import AvgPool2d
from torch.nn import CrossEntropyLoss
from torch.nn import UpsamplingBilinear2d
from torch.nn import Sigmoid
from torch.nn import ReLU6
from torch.nn import ReLU
from torch.nn import MSELoss
import torch.nn.functional as F
#from torch.nn import CosineSimilarity
from torch.optim import RMSprop
from torch.optim import Adam
import torch

eps = 1e-8

w, h = 512, 660
batch, size = 16, w*h
x_dim = w
loss_fn = MSELoss()
ones = Variable(torch.ones(batch))
zeros = Variable(torch.zeros(batch))

def xavier_init(size):
  stddev = 1./np.sqrt(size[0]/2)
  return Parameter(torch.randn(*size) * stddev, requires_grad=True)

dw1 = xavier_init([size, h])
db1 = Parameter(torch.zeros(h), requires_grad=True)
dw2 = xavier_init([h, 1])
db2 = Parameter(torch.zeros(1), requires_grad=True)

gw1 = xavier_init([size, h])
gb1 = Parameter(torch.zeros(h), requires_grad=True)
gw2 = xavier_init([h, size])
gb2 = Parameter(torch.zeros(size), requires_grad=True)

gen_params = [gw1, gb1, gw2, gb2]
dis_params = [dw1, db1, dw2, db2] 
global_params = gen_params + dis_params

class Discriminator(Module):

  def __init__(self):
    super(Discriminator, self).__init__()
#    self.seq = Sequential(
#      Conv2d(1, 32,
#        kernel_size=3,
#        stride=1,
#        padding=0,
#        bias=True),
#      MaxPool2d(kernel_size=2),
#      BatchNorm2d(32),
#      Sigmoid(),
#      Conv2d(32, 16,
#        kernel_size=3,
#        stride=2,
#        padding=0,
#        bias=True),
#      MaxPool2d(kernel_size=2),
#      BatchNorm2d(16),
#      Sigmoid(),
#      Conv2d(16, 8,
#        kernel_size=3,
#        stride=2,
#        padding=0,
#        bias=True),
#      MaxPool2d(kernel_size=2),
#      BatchNorm2d(8),
#      Sigmoid(),
#      Conv2d(8, 4,
#        kernel_size=2,
#        stride=1,
#        padding=0,
#        bias=True),
#      MaxPool2d(kernel_size=3),
#      BatchNorm2d(4),
#      Sigmoid(),
#      Conv2d(4, 1,
#        kernel_size=2,
#        stride=1,
#        padding=0,
#        bias=True),
#      MaxPool2d(kernel_size=2),
#      BatchNorm2d(1),
#      Sigmoid(),
#    )
#    self.linear = Sequential(
#        Linear(size, h),
#        ReLU(),
#        Linear(h, 1),
#        Sigmoid(),
#    )
    self.dw1 = xavier_init([size, h])
    self.db1 = Parameter(torch.zeros(h), requires_grad=True)
    self.dw2 = xavier_init([h, 1])
    self.db2 = Parameter(torch.zeros(1), requires_grad=True)
    
  def forward(self, x):
    #x = self.seq(x)
    #x = self.linear(x)
    t = F.relu(x @ self.dw1 + self.db1.repeat(x.size(0), 1))
    x = F.sigmoid(t @ self.dw2 + self.db2.repeat(t.size(0), 1))

    return x

class Generator(Module):

  def __init__(self):
    super(Generator, self).__init__()
#    self.seq = Sequential(
#      UpsamplingBilinear2d(scale_factor=2),
#      Conv2d(1, 32,
#        kernel_size=4,
#        stride=2,
#        padding=0,
#        bias=True),
#      Sigmoid(),
#      UpsamplingBilinear2d(scale_factor=2),
#      Conv2d(32, 16,
#        kernel_size=4,
#        stride=3,
#        padding=0,
#        bias=True),
#      Sigmoid(),
#      UpsamplingBilinear2d(scale_factor=2),
#      Conv2d(16, 1,
#        kernel_size=3,
#        stride=2,
#        padding=0,
#        bias=True),
#      Sigmoid(),
#      UpsamplingBilinear2d(scale_factor=3),
#      Conv2d(1, 1,
#        kernel_size=3,
#        stride=2,
#        padding=4,
#        bias=True),
#      Sigmoid(),
#    )
#    self.linear = Sequential(
#        Linear(size, h),
#        ReLU(),
#        Linear(h, size),
#        Sigmoid(),
#    )
    self.gw1 = xavier_init([size, h])
    self.gb1 = Parameter(torch.zeros(h), requires_grad=True)
    self.gw2 = xavier_init([h, size])
    self.gb2 = Parameter(torch.zeros(size), requires_grad=True)

  def forward(self, x):
    #x = self.seq(x)
    #x = self.linear(x)
    t = F.relu(x @ self.gw1 + self.gb1.repeat(x.size(0), 1))
    x = F.sigmoid(t @ self.gw2 + self.gb2.repeat(t.size(0), 1))
    return x

gen = Generator()
dis = Discriminator()
opt_gen = Adam(gen.parameters(), lr=1e-3)#chain(gen.parameters(), dis.parameters()), lr=1e-3)
opt_dis = Adam(dis.parameters(), lr=1e-3)#chain(gen.parameters(), dis.parameters()), lr=1e-3)

def cosine_similarity(x1, x2, dim=1, eps=1e-08):
  w12 = torch.sum(x1 * x2, 0)
  w1 = torch.norm(x1, 2)
  w2 = torch.norm(x2, 2)
  return (w12 / (w1*w2).clamp(min=eps)).squeeze()

def rand_sample(batch, size):
  return Variable(torch.randn(batch, size))

nor = transforms.Normalize(mean=[0.258, 0.248, 0.238],
                           std=[0.256, 0.233, 0.256])
        
def start():
  args = init()
  epochs = args.epochs
  gen_path = args.outpath

  if not isdir(gen_path):
    mkdir(gen_path)

  def get_x(data):
      data = data.astype(np.float32)
      data = torch.from_numpy(data)
      data = nor(data)
      X = Variable(data, volatile=False)
      return X

  def get_y(label):
      y = label.astype(np.int64)
      y = Variable(torch.from_numpy(y))
      return y

  gen_data = None
  model_path = '{}/{}'.format(args.model_root, args.model_id)

  for e in range(epochs):
    global_epoch = args.init_epoch+e+1
    for i, (data, y) in enumerate(data_generator(args.data_root, args.label_path)):
      if y.size == 0:
        continue

      true_y = get_y(y)
      data[np.where(data < 12000)] = 0.
      real_x = get_x(data)
      np.save('{}/{}'.format(gen_path, 0), real_x.data.numpy())
     
      ind = i % 16

      # discriminator
      real_y = dis(real_x)
      gen_data = rand_sample(batch, size)
      fake_x = gen(gen_data)
      fake_y = dis(fake_x)#get_x(fake_x.data.numpy()))
      dis_loss = optimize_dis(fake_y, real_y, true_y)

      # generator
      gen_data = rand_sample(batch, size)
      fake_x = gen(gen_data)
      fake_y = dis(fake_x)#get_x(fake_x.data.numpy()))
      gen_loss = optimize_gen(fake_y, real_y, true_y)

      # cross loss
      c_loss = Variable(torch.Tensor(1))
      #c_loss = cross_loss(fake_y, real_y, true_y)

      print('[{}/{}] dis_loss:{} gen_loss:{} c_loss:{}'.format(
            e+1, i+1, dis_loss.data.numpy(), gen_loss.data.numpy(), c_loss.data.numpy()), flush=True)

      ind and np.save('{}/{}'.format(gen_path, ind), gen(gen_data).data.numpy()) or None
      #np.save('{}/{}'.format(gen_path, ind), gen(gen_data.view(w,h)).data.numpy())

    torch.save({
          'epoch': global_epoch,
          'model': gen,
          }, '{}-gen-{}-{:.4f}.chkpt'.format(model_path, global_epoch, gen_loss.data.numpy()[0]))
    torch.save({
          'epoch': global_epoch,
          'model': dis,
          }, '{}-dis-{}-{:.4f}.chkpt'.format(model_path, global_epoch, dis_loss.data.numpy()[0]))

def optimize_dis(fake_y, real_y, y):
#  dis_loss = -(torch.mean(torch.log(real_y+eps)) + torch.mean(torch.log(1-fake_y+eps)))
  dis_loss = F.binary_cross_entropy(torch.squeeze(real_y+eps), ones) +  \
                F.binary_cross_entropy(torch.squeeze(fake_y+eps), zeros)
  dis_loss.backward()
  opt_dis.step()
  opt_dis.zero_grad()
  return dis_loss

def optimize_gen(fake_y, real_y, y):
#  gen_loss = -torch.mean(torch.log(fake_y+eps)+torch.log(1-real_y+eps))
#  gen_loss = loss_fn(fake_y, real_y)
  gen_loss = F.binary_cross_entropy(torch.squeeze(fake_y+eps), ones)
  gen_loss.backward()
  opt_gen.step()
  opt_gen.zero_grad()
  return gen_loss

def cross_loss(fake_y, real_y, y):
  c_loss = F.cross_entropy(real_y.view(batch), torch.squeeze(y)) + F.cross_entropy(fake_y.view(batch), torch.squeeze(y))
  return c_loss

if __name__ == '__main__':
  start()
  #plot_gan()

