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

train_data_source = "../data/russian_housing/train.csv"
eval_data_source = "../data/russian_housing/eval.csv"
test_data_source = "../data/russian_housing/test.csv"
chunksize = 32
max_features = 290   # remove id and price
model_id = 'russian_housing'
prep_data_path = '../build/{}-prep.csv'.format(model_id)
strep_log, rep_log, model_path, model_chkpt_path, tsboard_log = [None]*5
steps_per_epoch, init_epoch, total_epochs = 10, 0, 3
is_training, is_evaluating = False, False

def create_model():
  x_input = Input(shape=(max_features,), dtype='float32', name='input')
#  x = Reshape((max_features, 1))(x_input)
  x = BatchNormalization()(x_input)
  x = Dense(max_features, kernel_initializer='normal', activation='relu')(x)
#  x = Dropout(0.3)(x)

  y = Dense(1, kernel_initializer='normal', activation='softmax', name='output')(x)

  model = Model(inputs=[x_input], outputs=[y], name='rushs')
  #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc', 'sparse_categorical_accuracy', 'binary_accuracy'])
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'acc'])

  model.summary()
  plot_model(model, to_file='{}.png'.format(model_path), show_shapes=True, show_layer_names=True)
  print('model_path:{} steps:{} epochs:{}/{} chunksize:{} max_features:{}'.format(
       model_path, steps_per_epoch, init_epoch, total_epochs, chunksize, max_features))
  model.save(model_path)
  return model

def normalize(data):
  return (data-data.mean())/data.std()

def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'eval'])
  parser.add_argument('--prefix', default=model_id, type=str, help='Prefix for model persistance')
  parser.add_argument('--init_epoch', default=init_epoch, type=int, help='Initial epoch')
  parser.add_argument('--epochs', default=total_epochs, type=int, help='Total epoch to run')
  parser.add_argument('--steps', default=steps_per_epoch, type=int, help='Number of steps in one epoch')
  args = parser.parse_args()
  return args

def update_op_param(args):
  global total_epochs, init_epoch, steps_per_epoch
  total_epochs, init_epoch, steps_per_epoch = args.epochs, args.init_epoch, args.steps

def start(args):
  global is_training, is_evaluating
  update_path(args.prefix)
  update_op_param(args)

  if not isfile(prep_data_path):
    preprocess(train_data_source)

  if args.mode == 'train':
    is_traininig, is_evaluating = True, False
    train()
  elif args.mode == 'eval':
    is_traininig, is_evaluating = False, True
    evaluate()
  else:
    is_training, is_evaluating = False, False
    test()

def test():
  pass

def evaluate():
  pass

def get_model(model_path):
  model = load_model(model_path) if isfile(model_path) else create_model()
  print('name:{} lr:{} len(weights):{}'.format(model.name, K.eval(model.optimizer.lr), len(model.weights)))
  return model

def plot_history(history):
  keys = ['mse', 'loss',] #val_loss
  for k in keys:
    plt.plot(history.history[k])
  plt.legend(keys, loc='upper right')
  plt.title('model mse')
  plt.ylabel('mse/loss')
  plt.xlabel('epoch')
  plt.show()

def update_path(prefix):
  global model_id, model_path, model_chkpt_path, tsboard_log, strep_log, rep_log, prep_data_path
  model_id = prefix
  model_path = "../models/{}.h5".format(model_id)
  model_chkpt_path = "../models/"+model_id+"_chkpt_{epoch:02d}_{acc:.2f}.h5"
  tsboard_log = '../build/{}.log'.format(model_id)
  strep_log = '../build/log-{}.csv'.format(model_id)
  rep_log = '../build/log-{}.rep'.format(model_id)
  prep_data_path = '../build/{}-prep.csv'.format(model_id)

def train():
  model = get_model(model_path)
  chkpt = ModelCheckpoint(model_chkpt_path, monitor='mse', verbose=1)
  early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
  rep = StateReport(strep_log)
  tsboard = TensorBoard(tsboard_log)
  history = model.fit_generator(generate_data(prep_data_path), verbose=1,
        callbacks=[chkpt, early_stop, tsboard, rep],
        steps_per_epoch=steps_per_epoch, epochs=total_epochs, initial_epoch=init_epoch)
  model.save(model_path)
  plot_history(history)

def preprocess(source):
  reader = read_csv(source, header=0, chunksize=chunksize)
  if isfile(prep_data_path):
    unlink(prep_data_path)

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
    data.to_csv(prep_data_path) if not isfile(prep_data_path) else data.to_csv(prep_data_path, mode='a', header=False)

#plt.ion()
fig = plt.figure(figsize=(12, 12), facecolor='darkgray', edgecolor='black')
gs = gridspec.GridSpec(5, 7)
#fig.show()

def plot_data(col, i):
  i = i < 35 and i or i%35
  ax = plt.subplot(gs[i], facecolor='black')
  #ax = fig.add_subplot(max_features, i+1, i+1, facecolor='black')
  ax.autoscale(True)
  ax.plot(np.arange(len(col)), col, '.', color='b')
  fig.canvas.draw()

def load_chunk(chunk):
  chunk = chunk.fillna(0)
  ids, y = chunk['id'], chunk['price_doc'].values
  chunk['timestamp'] = chunk['timestamp'].values.astype('datetime64').astype(np.int)
  x = chunk.iloc[0:chunksize, 1:291].values #, 12:291]
  #for i in range(x.shape[1]):
  #  x[:, i] = normalize(x[:, i])
  y = normalize(y)
  #plot_data(y, 0)
  return x, y

def generate_data(source, forever=True): 
  while True:
    reader = read_csv(source, header=0, chunksize=chunksize)
    for chunk in reader:
      x, y = load_chunk(chunk)
      yield x, y
    if not forever: break

if __name__ == '__main__':
  args = init()
  start(args)
  #update_path(args.prefix)
  #update_op_param(args)
  #[x for x in generate_data(prep_data_path)]

  

