from apps.common import StateReport
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import dot
from keras.layers import Input
from keras.layers import Embedding
from keras.models import load_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.optimizers import SGD
#from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
import keras.backend as K
import keras
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import string
from os.path import isfile
import sys
import numpy as np
import glob
import _pickle

train_data_source = '../data/quora/origin-train.csv'
evaluate_data_source = '../data/quora/tiny-train.csv'
test_data_source = '../data/quora/test.csv'
test_result = '../data/quora/test_result.csv'
#model_path = "../models/quora_mlp.pkl"
tokenizer_path = "../models/quora_tokenizer.pkl"
model_id = 'quora_b'
model_path, model_chkpt_path, tsboard_log, logfd = [None]*4
max_features = 128
max_encoded_len = 128
batch_size = 64
steps_per_epoch = 4000
total_epochs = 1000
init_epoch = 16
learning_rate = 0.001
is_training = True
is_evaluating = False

def preprocess(q):
  q = [x.lower() for x in q]
  q = [''.join(c for c in x if c not in string.punctuation) for x in q]
  q = [''.join(c for c in x if c not in '0123456789') for x in q]
  q = [' '.join(x.split()) for x in q]
  return q

def create_model(tokenizer=None):
  ## with tfidf input
  feat_nr = len(tokenizer.word_counts)+1

  S, F = 30, 300
  x1_input = Input(shape=(max_features,), dtype='int32', name='x1_input')
  x2_input = Input(shape=(max_features,), dtype='int32', name='x2_input')

  x1 = Embedding(input_dim=feat_nr, output_dim=64, trainable=False)(x1_input)
  x2 = Embedding(input_dim=feat_nr, output_dim=64, trainable=False)(x2_input)

  x1 = BatchNormalization()(x1)
  x1 = Conv1D(128, 3, activation='relu')(x1)
  x1 = MaxPooling1D(3)(x1)

  x2 = BatchNormalization()(x2)
  x2 = Conv1D(128, 3, activation='relu')(x2)
  x2 = MaxPooling1D(3)(x2)

  x = dot([x1, x2], -1, normalize=True)
#  x = Conv1D(128, 3, activation='relu')(x)
#  x = MaxPooling1D(3)(x)
#  x = Conv1D(128, 3, activation='relu')(x)
#  x = MaxPooling1D(3)(x)
#  x = Conv1D(128, 3, activation='relu')(x)
#  x = MaxPooling1D(3)(x)
#  x = LSTM(32)(x)
  x = Flatten()(x)
  x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
  x = Dropout(0.3)(x)
  x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
  x = Dropout(0.3)(x)
  x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
  x = Dropout(0.3)(x)

  y = Dense(2, kernel_initializer='uniform', activation='softmax', name='output')(x)
  model = Model(inputs=[x1_input, x2_input], outputs=[y], name='final')
#  sgd = SGD(lr=learning_rate)
  opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.00000000001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc','sparse_categorical_accuracy', 'binary_accuracy'])
   
  model.summary()
  plot_model(model, to_file='{}.png'.format(model_path), show_shapes=True, show_layer_names=True)
  print('model_path:{} steps:{} epochs:{}/{} batch_size:{} max_features:{}'.format(
       model_path, steps_per_epoch, init_epoch, total_epochs, batch_size, max_features))
  return model

def process_data(data, tokenizer):
  q1, q2, labels = data['question1'].values.astype('U'),\
           data['question2'].values.astype('U'), None
  x1 = tokenizer.texts_to_sequences(q1)#, mode='binary') 
  x2 = tokenizer.texts_to_sequences(q2)#, mode='binary') 
  x1 = pad_sequences(x1, padding='post', truncating='post', dtype=int, maxlen=max_features)
  x2 = pad_sequences(x2, padding='post', truncating='post', dtype=int, maxlen=max_features)
  if is_training or is_evaluating:
    labels = data['is_duplicate'].values.astype(np.int32)
    labels = keras.utils.np_utils.to_categorical(labels, 2)
    return x1, x2, labels
  tid = data['test_id'].values.astype('U')
  return x1, x2, tid
 
def init():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'eval'])
  parser.add_argument('--train-tokenizer', dest='train_tokenizer', action='store_true', help='Pretrain tokenizer')
  parser.add_argument('--model_id', default='quora_b', type=str, help='Prefix for model persistance')
  parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
  parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
  parser.add_argument('--steps', default=1000, type=int, help='Number of steps in one epoch')
  parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
  parser.add_argument('--master', default='', type=str, help='Master for distributed execution')
  args = parser.parse_args()
  return args

def update_path(prefix):
  global model_path, model_chkpt_path, tsboard_log, logfd
  model_id = prefix
  model_chkpt_path = "../models/"+model_id+"_model_chkpt_{epoch:02d}-{acc:.2f}.h5"
  tsboard_log = '../build/{}.log'.format(model_id)
  # load last check point
  model_path = "../models/{}_model_chkpt_{}*.h5".format(model_id, init_epoch)
  chkpts = glob.glob(model_path)
  model_path = "../models/{}_model.h5".format(model_id) if len(chkpts) == 0 else chkpts[-1]

def update_model_params(args):
  global batch_size, steps_per_epoch, total_epochs, init_epoch, learning_rate
  batch_size = args.batch_size
  steps_per_epoch = args.steps
  total_epochs = args.epochs
  init_epoch = args.init_epoch
  learning_rate = args.lr

def start(args):
  global is_training, is_evaluating
  update_model_params(args)
  update_path(args.model_id)
  if args.mode == 'train':
    is_traininig, is_evaluating = True, False
    train(args.train_tokenizer)
  elif args.mode == 'eval':
    is_traininig, is_evaluating = False, True
    evaluate()
  else:
    is_training, is_evaluating = False, False
    test()

def pretrain_tokenizer(tokenizer, source, tokenizer_path=None):
  reader = pd.read_csv(source, header=0, chunksize=1000)
  print('-'*40)
  for data in reader:
    q1, q2, labels = data['question1'].values.astype('U'),\
           data['question2'].values.astype('U'),\
           data['is_duplicate'].values.astype(np.int32)
    tokenizer.fit_on_texts(np.concatenate((q1, q2)))
    print('tokenizer: word_count:{} word_docs:{} word_index:{} doc_count:{}'.format(
      len(tokenizer.word_counts), len(tokenizer.word_docs),
      len(tokenizer.word_index), tokenizer.document_count), end='\r')
  print('')
  if tokenizer_path is not None:
    _pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
  return tokenizer
   
def get_tokenizer(tokenizer_path=None, train=False, source=None):
  tokenizer = _pickle.load(open(tokenizer_path, 'rb')) if isfile(tokenizer_path) else Tokenizer()
  if train and source is not None:
    tokenizer = pretrain_tokenizer(tokenizer, source, tokenizer_path)
  print('using tokenizer: word_count:{} word_docs:{} word_index:{} doc_count:{}'.format(
    len(tokenizer.word_counts), len(tokenizer.word_docs),
    len(tokenizer.word_index), tokenizer.document_count))
  return tokenizer

def generate_data(source, tokenizer):
  if is_training or is_evaluating:
    while True:
      reader = pd.read_csv(source, header=0, chunksize=batch_size)
      for data in reader:
        x1, x2, y = process_data(data, tokenizer)
        yield {'x1_input': x1, 'x2_input': x2}, {'output': y}
  else:
    reader = pd.read_csv(source, header=0, chunksize=batch_size)
    for data in reader:
      x1, x2 = process_data(data, tokenizer)
      yield {'x1_input': x1, 'x2_input': x2}

def read_data(source, tokenizer):
  data = pd.read_csv(source, header=0)
  x1, x2, y = process_data(data, tokenizer)
  return x1, x2, y

def get_model(model_path, tokenizer=None):
  model = load_model(model_path) if isfile(model_path) else create_model(tokenizer)
  K.set_value(model.optimizer.lr, 0.001)
  K.set_value(model.optimizer.decay, 0.0)
  print('name:{} lr:{} decay:{} len(weights):{}'.format(
    model.name,
    K.eval(model.optimizer.lr),
    K.eval(model.optimizer.decay),
    len(model.weights)))
  return model

def plot_history(history):
  keys = ['acc', 'loss',] #val_loss
  for k in keys:
    plt.plot(history.history[k])
  plt.legend(keys, loc='upper right')
  plt.title('model accuracy')
  plt.ylabel('accuracy/loss')
  plt.xlabel('epoch')
  plt.show()

def train(train_tokenizer=False):
  tokenizer = get_tokenizer(tokenizer_path, source=train_data_source, train=train_tokenizer)
  model = get_model(model_path, tokenizer)
  chkpt = ModelCheckpoint(model_chkpt_path, monitor='acc', verbose=1)
  early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
  rep = StateReport('../build/log-{}.csv'.format(model_id))
  tsboard = TensorBoard(tsboard_log)
  history = model.fit_generator(generate_data(train_data_source, tokenizer), callbacks=[chkpt, early_stop, rep, tsboard],\
          verbose=1, steps_per_epoch=steps_per_epoch, epochs=total_epochs, initial_epoch=init_epoch)# workers=4, pickle_safe=True)
#  history = model.fit({'x1_input':x1, 'x2_input':x2}, y, nb_epoch=total_epochs, batch_size=batch_size, verbose=1, evaluate_split=0.1)
  model.save(model_path)
  plot_history(history)

def evaluate():
  global logfd
  # TODO: split train data
  tokenizer = get_tokenizer(tokenizer_path, source=train_data_source, train=False)
  model = get_model(model_path, tokenizer)
  scalars = model.evaluate_generator(generate_data(train_data_source, tokenizer), steps=steps_per_epoch)
  print('total scalars:{}'.format(len(scalars)))
  [print("{}:{}".format(m, s)) for m, s in zip(model.metrics_names, scalars)]

def test():
  if not isfile(model_path):
    print('No model found @ {}'.format(model_path))
    return
  tokenizer = get_tokenizer(tokenizer_path, source=train_data_source, train=False)
  model = get_model(model_path, tokenizer)
  batch_size = 55
  reader = pd.read_csv(test_data_source, header=0, chunksize=batch_size)
  with open(test_result, 'w+') as fd:
    fd.write('test_id,is_duplicate\n')
    for chunk in reader:
      x1, x2, tid = process_data(chunk, tokenizer)
      x = {'x1_input': x1, 'x2_input': x2}
      res = model.predict(x, batch_size=batch_size, verbose=0)
      res = res.argmax(1)
      [fd.write('{},{}\n'.format(c, r)) for c, r in zip(tid, res)]

if __name__ == '__main__':
  args = init()
  start(args)

