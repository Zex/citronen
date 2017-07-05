from apps.common import StateReport
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.gridspec as gridspec
from pandas import read_csv
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras import regularizers
from keras.layers import Input
from keras.utils import plot_model
from keras.layers.core import Dense
from keras.layers.core import Reshape
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers import LSTM
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.preprocessing.text import one_hot
from keras.initializers import TruncatedNormal
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
from os.path import isfile
from os import unlink
import argparse

class RusKr(object):
  modes = ['train', 'eval', 'test']

  def __init__(self, args):
    self.model_id = args.model_id
    self.update_path()
    self.batch_size = args.batch_size
    self.max_features = 290
    self.lr = args.lr
    self.mode = args.mode
    self.init_epoch = args.init_epoch
    self.total_epochs = args.epochs
    self.steps_per_epoch = args.steps
    self.master = args.master
    self.global_step = 0
    self.learning_rate = args.lr
    [setattr(self, "is_{}".format(mode), self.mode==mode) for mode in RusKr.modes]

  def update_path(self):
    self.model_path = "../models/{}.h5".format(self.model_id)
    self.chkpt_path = "../models/{}.chkpt".format(self.model_id)
    self.logdir = "../build/{}.log".format(self.model_id)
    self.strep_log = "../build/{}_step_report.csv".format(self.model_id)
    self.prep_data_path = '../build/{}-prep.csv'.format(self.model_id)
    self.data_base = '../data/russian_housing'
    self.train_data_path = "{}/train.csv".format(self.data_base)
    self.eval_data_path = "{}/eval.csv".format(self.data_base)
    self.test_data_path = "{}/test.csv".format(self.data_base)

  def create_model(self):
    x_input = Input(shape=(self.max_features,), dtype='float32', name='input')
    x = BatchNormalization()(x_input)
#    x = Reshape((self.batch_size, self.max_features))(x_input)
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = Flatten()(x)
    x = Dense(290, kernel_initializer='normal', activation='relu')(x)
    y = Dense(1, kernel_initializer='normal', activation='softmax')(x)
    self.model = Model(inputs=[x_input], outputs=[y], name=self.model_id)
    self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'acc'])
  
    self.model.summary()
    plot_model(self.model, to_file='{}.png'.format(self.model_path), show_shapes=True, show_layer_names=True)
    print('model_path:{} steps:{} epochs:{}/{} batch_size:{} max_features:{}'.format(
         self.model_path, self.steps_per_epoch, self.init_epoch, self.total_epochs, self.batch_size, self.max_features))
    self.model.save(self.model_path)
    return self.model

  def get_model(self):
    model = load_model(self.model_path) if isfile(self.model_path) else self.create_model()
    print('name:{} lr:{} len(weights):{}'.format(model.name, K.eval(model.optimizer.lr), len(model.weights)))
    return model

  def train(self):
    model = self.get_model()
    chkpt = ModelCheckpoint(self.chkpt_path, monitor='loss', verbose=1)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
    rep = StateReport(self.strep_log)
    tsboard = TensorBoard(self.logdir)
    history = model.fit_generator(generate_data(self.prep_data_path, self.batch_size), verbose=1,
          callbacks=[chkpt, early_stop, tsboard, rep],
          steps_per_epoch=self.steps_per_epoch, epochs=self.total_epochs, initial_epoch=self.init_epoch)
    model.save(self.model_path)
    plot_history(history)

  def eval(self):
    pass

  def test(self):
    pass

def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=RusKr.modes)
  parser.add_argument('--model_id', default='ruskr', type=str, help='Prefix for model persistance')
  parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
  parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
  parser.add_argument('--steps', default=1000, type=int, help='Number of steps in one epoch')
  parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
  parser.add_argument('--master', default='', type=str, help='Master for distributed execution')
  args = parser.parse_args()
  return args

def start(args):
#  setup_plot()
  rus = RusKr(args)
  print(rus.is_train)
  if not isfile(rus.prep_data_path):
    preprocess(rus.train_data_path, rus.prep_data_path, rus.batch_size)

  if rus.is_train:
    rus.train()
  elif rus.is_eval:
    rus.eval()
  else:
    test()

def plot_history(history):
  keys = ['acc', 'loss',] #val_loss
  for k in keys:
    plt.plot(history.history[k])
  ax1.legend(keys, loc='upper right')
  #plt.title('model mse')
  #plt.ylabel('mse/loss')
  #plt.xlabel('epoch')
  #plt.show()
  ax1.plot(np.arange(len(y)), y)
  fig.canvas.draw()

def preprocess(source, dest, batch_size=64):
  reader = read_csv(source, header=0, chunksize=batch_size)

  for data in reader:
    data = data.fillna(0)
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    timestamp = data['timestamp'].values.astype('datetime64').astype(np.int)
    product_type, sub_area, ecology = \
      data['product_type'].values, data['sub_area'].values, data['ecology'].values
    data['timestamp'] = timestamp
    data['product_type'] = np.reshape([one_hot(
          x, n=np.unique(product_type).shape[0]+1, filters='') for x in product_type], product_type.shape)
    sub_area = np.array([s.replace(' ', '').replace('-', '').replace('\'', '').replace(',', '') for s in sub_area])
    data['sub_area'] = np.reshape([one_hot(
      x, n=np.unique(sub_area).shape[0]+1) for x in sub_area], sub_area.shape)
    ecology = np.array([s.replace(' ', '').replace('-', '').replace('\'', '').replace(',', '') for s in ecology])
    data['ecology'] = np.reshape([one_hot(
      x, n=np.unique(ecology).shape[0]+1) for x in ecology], ecology.shape)
    data.to_csv(dest) if not isfile(dest) else data.to_csv(dest, mode='a', header=False)
    #ids, y = data['id'], data['price_doc'].values
    #ax0.plot(np.arange(len(y)), y)
    #fig.canvas.draw()

def setup_plot():
  global fig, ax0, ax1
  plt.ion()
  fig = plt.figure(figsize=(12, 8), facecolor='darkgray', edgecolor='black')
  ax0 = fig.add_subplot(211, facecolor='black')
  ax0.autoscale(True)
  ax1 = fig.add_subplot(212, facecolor='black')
  ax1.autoscale(True)
  fig.show()

def plot_data(col, i):
  i = i < 35 and i or i%35
  ax = plt.subplot(gs[i], facecolor='black')
  #ax = fig.add_subplot(max_features, i+1, i+1, facecolor='black')
  ax.autoscale(True)
  ax.plot(np.arange(len(col)), col, '.', color='b')
  fig.canvas.draw()

def load_data(data, batch_size=64):
  data = data.fillna(0)
  ids, y = data['id'], data['price_doc'].values
  data['timestamp'] = data['timestamp'].values.astype('datetime64').astype(np.int)
  x = data.iloc[0:batch_size, 1:291].values #, 12:291]
  #for i in range(x.shape[1]):
  #  x[:, i] = normalize(x[:, i])
  # y = normalize(y)
  ids, y = data['id'], data['price_doc'].values
  #ax0.plot(np.arange(len(y)), y)
  #fig.canvas.draw()
  #plot_data(y, 0)
  return x, y

def generate_data(source, batch_size=64, forever=True): 
  while True:
    reader = read_csv(source, header=0, chunksize=batch_size)
    for chunk in reader:
      x, y = load_data(chunk, batch_size)
      yield x, y
    if not forever: break

if __name__ == '__main__':
  args = init()
  start(args)

  

