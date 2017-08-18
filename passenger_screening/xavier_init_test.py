import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sn

def xavier_init(size):
  stddev = 1./np.sqrt(size[0]/2)
  return Variable(torch.randn(*size) * stddev, requires_grad=True)

def simple_init(size):
  return Variable(torch.randn(*size), requires_grad=True)

init_w = xavier_init([32, 32])
print(init_w.data.numpy().shape)
print(init_w.data.numpy())

#sn.plt.plot(init_w[0], init_w[1])
sn.plt.ion()
fig = sn.plt.figure(figsize=(8,8), facecolor='black', edgecolor='black')
fig.show()

gs = gridspec.GridSpec(2,2, wspace=1e-2, hspace=1e-20)
ax1 = fig.add_subplot(gs[0])
ax1.imshow(init_w.data.numpy())

ax2 = fig.add_subplot(gs[1])
for line in init_w.data.numpy():
  ax2.plot(np.arange(len(line)), line)

simple_w = simple_init([32, 32])
ax3 = fig.add_subplot(gs[2])
ax3.imshow(simple_w.data.numpy())

ax4 = fig.add_subplot(gs[3])
for line in simple_w.data.numpy():
  ax4.plot(np.arange(len(line)), line)

input('...')



