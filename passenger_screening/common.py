import argparse

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
