import matplotlib
matplotlib.use("TkAgg")
from common import init, plot_img, init_axs, data_generator, reinit_plot
import glob
import numpy as np
from os import mkdir
from os.path import isdir
import seaborn as sn
from torchvision import transforms
from torch.autograd import Variable
from torch import nn
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import AvgPool2d
from torch.nn import CrossEntropyLoss
from torch.nn import UpsamplingBilinear2d
from torch.nn import LeakyReLU
from torch.nn import Sigmoid
from torch.nn import Tanh
from torch.nn import ReLU
import torch.nn.functional as F
#from torch.nn import CosineSimilarity
from torch.optim import RMSprop
import torch


class Discriminator(Module):

  def __init__(self):
    super(Discriminator, self).__init__()
    self.seq = Sequential(
      Conv2d(1, 32,
        kernel_size=5,
        stride=3,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(32),
      ReLU(),
      Conv2d(32, 16,
        kernel_size=5,
        stride=2,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(16),
      ReLU(),
      Conv2d(16, 1,
        kernel_size=4,
        stride=2,
        padding=0,
        bias=False),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(1),
      ReLU(),
      Conv2d(1, 1,
        kernel_size=2,
        stride=1,
        padding=0,
        bias=False),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(1),
      LeakyReLU(),
    )
    self.linear = Sequential(
        Linear(660, 512),
        ReLU(),
        Linear(512, 1),
        LeakyReLU(),
    )

  def forward(self, x):
    x = self.seq(x)
    #x = self.linear(x)
    return x


class Generator(Module):

  def __init__(self):
    super(Generator, self).__init__()
    self.seq = Sequential(
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(1, 32,
        kernel_size=4,
        stride=2,
        padding=0,
        bias=True),
      ReLU(),
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(32, 16,
        kernel_size=4,
        stride=3,
        padding=0,
        bias=True),
      ReLU(),
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(16, 1,
        kernel_size=3,
        stride=2,
        padding=0,
        bias=True),
      ReLU(),
      UpsamplingBilinear2d(scale_factor=3),
      Conv2d(1, 1,
        kernel_size=3,
        stride=2,
        padding=4,
        bias=True),
      ReLU(),
    )
    self.linear = Sequential(
        Linear(660, 512),
        ReLU(),
        Linear(512, 660),
        LeakyReLU(),
    )

  def forward(self, x):
    x = self.linear(x)
    #x = self.seq(x)
    return x


gen = Generator()
dis = Discriminator()
opt_gen = RMSprop(gen.parameters())
opt_dis = RMSprop(dis.parameters())
nor = transforms.Normalize(mean=[0.254, 0.328, 0.671],
                            std=[0.229, 0.224, 0.225])
#loss_fn = CrossEntropyLoss()
#sim_fn = CosineSimilarity(dim=4)
eps = 1e-8

def cosine_similarity(x1, x2, dim=1, eps=1e-08):
  w12 = torch.sum(x1 * x2, 0)
  w1 = torch.norm(x1, 2)
  w2 = torch.norm(x2, 2)
  return (w12 / (w1*w2).clamp(min=eps)).squeeze()

def rand_sample(gen_data, expect, w, h):
  if len(gen_data) <= expect:
    #gen_data.append(Variable(torch.from_numpy(np.random.multinomial(1, [0.1], (660, 512)).T.reshape(w, h))).float())
    #gen_data.append(Variable(torch.from_numpy((np.ones((w, h))).reshape(1, 1, w, h)).float()))
    gen_data.append(Variable(torch.randn(w, h)))
  else:
    #gen_data[expect] = Variable(torch.from_numpy(np.random.multinomial(1, [0.1], (660, 512)).T.reshape(w, h))).float()
    gen_data[expect] = Variable(torch.randn(w, h))
        
def start():
  args = init()
  epochs = args.epochs
  w, h = 512, 660
  gen_path = "./gan_output"

  if not isdir(gen_path):
    mkdir(gen_path)

  def get_x(data):
      data = data.astype(np.float32)
      data = torch.from_numpy(data).contiguous().view(1, 1, 512, 660)
      #X = nor(data)
      X = Variable(data, volatile=False)
      return X

  def get_y(label):
      y = label.astype(np.int64)
      y = Variable(torch.from_numpy(y), volatile=False)
      return y

  gen_data = []
  model_path = '{}/{}'.format(args.model_root, args.model_id)

  for e in range(epochs):
    global_epoch = args.init_epoch+e+1
    for i, (data, y) in enumerate(data_generator(args.data_root, args.label_path)):
      if y.size == 0:
        continue
        
      true_y = get_y(y)
      real_x = get_x(data)

      #print('[{}] [dis] true_y:{}'.format(e, true_y.data.numpy()))
      # discriminator
      real_y = dis(real_x)
      #print('[{}] [dis] real_y:{}'.format(e, real_y.data.numpy()))

      ind = i%16
      rand_sample(gen_data, ind, w, h)
      fake_x = gen(gen_data[ind])
      #print('[{}] [dis] fake_x:{}'.format(e, fake_x.data.numpy().shape))
      gen_data[ind] = fake_x
      fake_y = dis(fake_x.view(1,1,w,h))
      #print('[{}] [dis] fake_y:{}'.format(e, fake_y.data.numpy()))
      dis_loss = optimize_dis(fake_y, real_y, true_y)

      # generator
      rand_sample(gen_data, ind, w, h)
      fake_x = gen(gen_data[ind])
      #print('[{}] [gen] fake_x'.format(e, fake_x.data.numpy().shape))
      fake_y = dis(fake_x.view(1,1,w,h))
      #print('[{}] [gen] fake_y:{}'.format(e, fake_y.data.numpy()))
      gen_loss = optimize_gen(fake_y, real_y, true_y)

      # cross loss
      c_loss = Variable(torch.Tensor(1))
      c_loss = cross_loss(fake_y, real_y, true_y)
      print('[{}] dis_loss:{} gen_loss:{} c_loss:{}'.format(e, dis_loss.data.numpy(), gen_loss.data.numpy(), c_loss.data.numpy()), flush=True)

      np.save('{}/{}'.format(gen_path, ind), gen(gen_data[ind]).data.numpy())

    torch.save({
          'epoch': global_epoch,
          'model': gen,
          }, '{}-gen-{}-{:.4f}.chkpt'.format(model_path, global_epoch, gen_loss.data.numpy()[0]))
    torch.save({
          'epoch': global_epoch,
          'model': dis,
          }, '{}-dis-{}-{:.4f}.chkpt'.format(model_path, global_epoch, dis_loss.data.numpy()[0]))

def optimize_dis(fake_y, real_y, y):
  opt_dis.zero_grad()
  #dis_loss = -torch.mean(torch.log(real_y + eps) + torch.log(1. - fake_y + eps))
  dis_loss = -(torch.mean(real_y) - torch.mean(fake_y))
  #dis_loss = torch.log(-torch.mean(real_y) + torch.mean(fake_y) + eps)
  dis_loss.backward(retain_variables=True)
  opt_dis.step()
  return dis_loss

def optimize_gen(fake_y, real_y, y):
  opt_gen.zero_grad()
  #gen_loss = -torch.mean(torch.log(fake_y + eps))
  gen_loss = -torch.mean(fake_y)
  gen_loss.backward()
  opt_gen.step()
  return gen_loss

def cross_loss(fake_y, real_y, y):
  print(fake_y.data.numpy().shape, real_y.data.numpy().shape)
  #y = y.repeat(1)
  c_loss = F.cross_entropy(real_y.view(1, 2), y) + F.cross_entropy(fake_y.view(1, 2), y)
  return c_loss

if __name__ == '__main__':
  start()
  #plot_gan()

