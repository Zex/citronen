from read_aps import read_header, read_data
import torch
from torch import from_numpy
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import CrossEntropyLoss
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
    self.conv1 = Conv2d(1, 64,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=True)
    self.max_pool1 = MaxPool2d(
                        kernel_size=3,
                        )
    self.conv2 = Conv2d(64, 128,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=True)
    self.max_pool2 = MaxPool2d(
                        kernel_size=3,
                        )
    self.conv3 = Conv2d(128, 256,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=True)
    self.max_pool3 = MaxPool2d(
                        kernel_size=3,
                        )

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.max_pool1(x)
    print('pool1', x.dim())
    x = F.relu(self.conv2(x))
    x = self.max_pool2(x)
    x = F.relu(self.conv3(x))
    x = self.max_pool3(x)
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


def data_generator(data_root):
  for src in glob.glob(data_root+'/*.aps'):
    print('current', src)
    header = read_header(src)
    data, _ = read_data(src, header)
    yield data, src

def load_labels(label_path):
  # Columns:
  #  - Id,Probability
  if not isfile(label_path):
    raise FileNotFoundError(label_path)
  labels = read_csv(label_path, header=0)  
  return labels

def start():
  args = PassengerScreening.init()
  chkpt_path = '{}/{}.chkpt'.format(args.model_root, args.model_id)
  # setup
  model = PassengerScreening()
  loss_fn = CrossEntropyLoss().cpu()
  optimizer = SGD(model.parameters(), args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

  for child in model.children():
    print('child:', child)
  # begin
  data_root = args.data_root
  labels = load_labels(args.label_path)
  try:
    for i, (data, src) in enumerate(data_generator(data_root)):
      iid = basename(src).split('.')[0]
      print(i, data.shape, src, iid)
      with Supervisor():
        for i in range(data.shape[2]):
          lid = iid + '_Zone' + str(i+1)
          fr, y = data[:,:,i], labels[labels['Id'] == lid]['Probability']
          fr = fr.astype(np.float32)
          fr = torch.from_numpy(fr).contiguous().view(1, 1, 512, 660)
          X = Variable(fr, volatile=True)
          y = y.astype(np.int32)
          y = torch.from_numpy(y.values)
          y = Variable(y, volatile=True)
          step(model, optimizer, loss_fn, X, y)
      torch.save({
          'epoch': i+1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          }, chkpt_path)
  except Exception as ex:
    print('epoch failed:', ex)
    raise

def step(model, optimizer, loss_fn, data, label):
  try:
    output = model(data)
    print('output', output)
    optimizer.zero_grad()
    print('opt', optimizer)
    loss_fn(output, label).backward() 
    optimizer.step()
  except Exception as ex:
    print('step failed:', ex)
    raise


if __name__ == '__main__':
  start()

