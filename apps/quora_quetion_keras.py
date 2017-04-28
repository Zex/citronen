import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import string
from os.path import isfile
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.layers.core import Flatten
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Merge
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
import keras.backend as K

shrink_data_source = '../data/quora/train_shrink.csv'
train_data_source = '../data/quora/train.csv'
validation_data_source = '../data/quora/validation.csv'
test_data_source = '../data/quora/test.csv'
test_result = '../data/quora/test_result.csv'
#model_path = "../models/quora_mlp.pkl"
model_path = "../models/quora_model.h5"
model_chkpt_path = "../models/quora_chkpt_{epoch:02d}-{acc:.2f}.h5"
max_features = 400
chunksize = 16
learning_rate = 0.001
max_epochs = 1000
token = Tokenizer()

def preprocess(q):
    q = [x.lower() for x in q]
    q = [''.join(c for c in x if c not in string.punctuation) for x in q]
    q = [''.join(c for c in x if c not in '0123456789') for x in q]
    q = [' '.join(x.split()) for x in q]
    return q

def create_model():
    x1_input = Input(shape=(max_features,), dtype='float32', name='x1_input')
    x2_input = Input(shape=(max_features,), dtype='float32', name='x2_input')

    x1 = Embedding(output_dim=max_features, input_dim=chunksize, input_length=max_features)(x1_input)
    x1 = LSTM(512, return_sequences=True, name='x1_lstm1')(x1)
    x1 = LSTM(128, return_sequences=True, name='x1_lstm2')(x1)
    x1 = LSTM(64, return_sequences=True, name='x1_lstm3')(x1)

    x2 = Embedding(output_dim=max_features, input_dim=chunksize, input_length=max_features)(x2_input)
    x2 = LSTM(512, return_sequences=True, name='x2_lstm1')(x2)
    x2 = LSTM(128, return_sequences=True, name='x2_lstm2')(x2)
    x2 = LSTM(64, return_sequences=True, name='x2_lstm3')(x2)

    print('x1.ndim:{}, x2.ndim:{}'.format(K.ndim(x1), K.ndim(x2)))
    x = concatenate([x1, x2])
    print('x.ndim:{}'.format(K.ndim(x)))
    x = LSTM(32, return_sequences=True, name='x_lstm1')(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(100, kernel_initializer='uniform', activation='relu', name='x_relu')(x)
    print('x.ndim:{}'.format(K.ndim(x)))
    y = Dense(1, kernel_initializer='uniform', activation='softmax', name='output')(x)
    print('y.ndim:{}'.format(K.ndim(y)))
    model = Model(inputs=[x1_input, x2_input], outputs=[y],  name='final')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.summary()
    return model

def process_data(data):
    # process train data
    q1, q2, labels = data['question1'].values.astype('U'),\
                     data['question2'].values.astype('U'),\
                     data['is_duplicate'].values.astype('U').astype(int)
    token.fit_on_texts(q1)#np.concatenate((q1, q2)))
    token.fit_on_texts(q2)
    x1 = token.texts_to_matrix(q1, mode='tfidf')  
    x2 = token.texts_to_matrix(q2, mode='tfidf')
    x1 = pad_sequences(x1, padding='post', truncating='post', dtype=float, maxlen=max_features)
    x2 = pad_sequences(x2, padding='post', truncating='post', dtype=float, maxlen=max_features)
#    x1 = x1.reshape((1, *x1.shape))
#    x2 = x2.reshape((1, *x2.shape))
    return x1, x2, labels
  
def do_train(model, q1_train, q2_tain, labels, q1_validation, q2_validation, labels_validation):
    # create model
    if model is None:
        model = create_model() 
    # train with data
    model.fit([q1_train, q2_train], labels, 
              validation_data=([q1_validation, q2_validation], labels_validation),
              verbose=1, batch_size=chunksize, nb_epoch=max_epochs)
    return model

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    args = parser.parse_args()
    return args

def start(args):
    if args.mode == 'train':
        train()
    else:
        test(model)

def generate_data():
    reader = pd.read_csv(train_data_source, header=0, chunksize=chunksize)
    for data in reader:
        x1, x2, y = process_data(data)
        yield {'x1_input': x1, 'x2_input': x2}, {'output': y}

def get_model():
    model = None
    if isfile(model_path):
        model = load_model(model_path)
    if model is None:
        model = create_model() 
    return model

def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['acc', 'loss'], loc='upper right')
    plt.show()

def train():
    model = get_model()
    chkpt = ModelCheckpoint(model_chkpt_path, monitor='loss', verbose=1)
    history = model.fit_generator(generate_data(), callbacks=[chkpt], verbose=1, steps_per_epoch=1000, epochs=10)
    model.save(model_path)
    plot_history(history)

def test():
    if not isfile(model_path):
        print('No model found @ {}'.format(model_path))
        return
    model = get_model()
    # data loader
    reader = pd.read_csv(test_data_source, header=0, chunksize=chunksize)
    # do test
    with open(test_result, 'w+') as fd:
        fd.write('test_id,is_duplicate\n')
        for data in reader:
            result = do_test(model, data)
            [fd.write('{},{}\n'.format(r[0], r[1])) for r in result]

if __name__ == '__main__':
    args = init()
    start(args)

