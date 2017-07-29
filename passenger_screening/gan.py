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
from torch.nn import Tanh
import torch.nn.functional as F
#from torch.nn import CosineSimilarity
from torch.optim import RMSprop
import torch


class Discriminator(Module):

  def __init__(self):
    super(Discriminator, self).__init__()
    self.seq = Sequential(
      Conv2d(1, 128,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(128),
      Tanh(),
      Conv2d(128, 64,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(64),
      Tanh(),
      Conv2d(64, 32,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=2),
      BatchNorm2d(32),
      Tanh(),
      Conv2d(32, 1,
        kernel_size=3,
        stride=2,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=3),
      BatchNorm2d(1),
      Tanh(),
      Conv2d(1, 1,
        kernel_size=3,
        stride=2,
        padding=0,
        bias=True),
      MaxPool2d(kernel_size=3),
      BatchNorm2d(1),
      Tanh(),
    )

  def forward(self, x):
    x = self.seq(x)
    #x = torch.max(x)
    return x


class Generator(Module):

  def __init__(self):
    super(Generator, self).__init__()
    self.seq = Sequential(
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(1, 128,
        kernel_size=4,
        stride=2,
        padding=0,
        bias=True),
      Tanh(),
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(128, 64,
        kernel_size=4,
        stride=3,
        padding=0,
        bias=True),
      Tanh(),
      UpsamplingBilinear2d(scale_factor=2),
      Conv2d(64, 1,
        kernel_size=3,
        stride=2,
        padding=0,
        bias=True),
      Tanh(),
      UpsamplingBilinear2d(scale_factor=3),
      Conv2d(1, 1,
        kernel_size=3,
        stride=2,
        padding=4,
        bias=True),
      Tanh(),
    )

  def forward(self, x):
    x = self.seq(x)
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
  print(w12.data.numpy().shape)
  print(w1.data.numpy())
  print(w2.data.numpy())
  print('clamp',  (w1*w2).clamp(min=eps))
  return (w12 / (w1*w2).clamp(min=eps)).squeeze()

def start():
  args = init()
  epochs = 10
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
      y = torch.from_numpy(y)
      y = Variable(y, volatile=False)
      return y

  gen_data = []
  model_path = '{}/{}'.format(args.model_root, args.model_id)

  for e in range(epochs):
    global_epoch = args.init_epoch+e+1
    for i, (data, y) in enumerate(data_generator(args.data_root, args.label_path)):
      if i > 32: # exp
        break
      true_y = get_y(y)
      real_x = get_x(data)

      real_y = dis(real_x)
      print('[dis] output', real_y.data.numpy().shape)

      if len(gen_data) < i+1:
        gen_data.append(Variable(torch.from_numpy((np.random.randn(w, h)*10000).reshape(1, 1, w, h)).float()))
      fake_x = gen(gen_data[i])
      print('[gen] output', fake_x.data.numpy().shape)
      gen_data[i] = fake_x.data.numpy()

      fake_y = dis(fake_x)
      print('[dis] gen', fake_y.data.numpy().shape)
      dis_loss, gen_loss, total = optimize(fake_x, real_x, fake_y, real_y, true_y)

    [np.save('{}/{}'.format(gen_path, i), data) for i, data in enumerate(gen_data)]
    torch.save({
          'epoch': global_epoch,
          'model': gen,
          }, '{}-gen-{}-{:.4f}.chkpt'.format(model_path, global_epoch, gen_loss))
    torch.save({
          'epoch': global_epoch,
          'model': gen,
          }, '{}-dis-{}-{:.4f}.chkpt'.format(model_path, global_epoch, dis_loss))

def optimize(fake_x, real_x, fake_y, real_y, y):
#  print(real_x.data.numpy().shape, real_x.data.numpy().shape)

  dis_loss = torch.mean(torch.log(real_x + eps) + torch.log(1. - fake_x + eps))
  y = y.repeat(1)
  gen_loss = -F.cross_entropy(real_y.view(1,2), y) - F.cross_entropy(fake_y.view(1,2), y)

  total_loss = -(dis_loss + gen_loss)

  total_loss.backward()
  print('dis_loss:{} gen_loss:{} total:'.format(dis_loss.data.numpy(), gen_loss.data.numpy(), total_loss.data.numpy()))

  opt_dis.zero_grad()
  opt_dis.step()

  opt_gen.zero_grad()
  opt_gen.step()

  return dis_loss, gen_loss, total_loss

def plot_gan():
  gen_path = "./gan_output"
  sn.plt.ion()
  fig = sn.plt.figure()
  ax = fig.add_subplot(111)
  fig.show()
  for i in glob.iglob('{}/*'.format(gen_path)):
    with open(i, 'rb') as fd:
      fd.seek(256)
      buf = np.fromfile(fd, dtype=np.float32)
      buf = buf.reshape(252, 326)
      ax.imshow(buf)

def save_buffer(data, path):
  with open(path, 'wb') as f:
    f.write(data)


if __name__ == '__main__':
  start()
  #plot_gan()

