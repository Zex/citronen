from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
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
#import keras.backend as K
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
import _pickle

train_data_source = '../data/quora/origin-train.csv'
validation_data_source = '../data/quora/validation.csv'
test_data_source = '../data/quora/test.csv'
test_result = '../data/quora/test_result.csv'
#model_path = "../models/quora_mlp.pkl"
tokenizer_path = "../models/quora_tokenizer.pkl"
model_id = 'quora_extra'
model_path = "../models/{}_model.h5".format(model_id)
model_chkpt_path = "../models/"+model_id+"_model_chkpt_{epoch:02d}-{acc:.2f}.h5"
tsboard_log = '../build/{}.log'.format(model_id)
logfd = open('../build/log-{}.csv'.format(model_id), 'w+')
max_features = 128
max_encoded_len = 128
chunksize = 64
steps_per_epoch = 4000
total_epochs = 60
init_epoch = 40
learning_rate = 0.001

class PlotLog(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        logfd.write('{},{},{},{}\n'.format(logs.get('loss'), logs.get('acc'), logs.get('categorical_accuracy'), logs.get('sparse_categorical_accuracy')))
        logfd.flush()

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
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = Conv1D(128, 3, activation='relu')(x)
#    x = MaxPooling1D(3)(x)
#    x = LSTM(32)(x)
    x = Flatten()(x)
    x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, kernel_initializer='uniform', activation='relu')(x)
    x = Dropout(0.2)(x)

    y = Dense(2, kernel_initializer='uniform', activation='softmax', name='output')(x)
    model = Model(inputs=[x1_input, x2_input], outputs=[y],  name='final')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['loss', 'acc','sparse_categorical_accuracy'])
   
    model.summary()
    plot_model(model, to_file='{}.png'.format(model_path), show_shapes=True, show_layer_names=True)
    print('model_path:{} steps:{} epochs:{}/{} chunksize:{} max_features:{}'.format(
           model_path, steps_per_epoch, init_epoch, total_epochs, chunksize, max_features))
    return model

def process_data(data, tokenizer):
    q1, q2, labels = data['question1'].values.astype('U'),\
                     data['question2'].values.astype('U'),\
                     data['is_duplicate'].values.astype(np.int32)
    x1 = tokenizer.texts_to_sequences(q1)#, mode='binary') 
    x2 = tokenizer.texts_to_sequences(q2)#, mode='binary') 
    x1 = pad_sequences(x1, padding='post', truncating='post', dtype=int, maxlen=max_features)
    x2 = pad_sequences(x2, padding='post', truncating='post', dtype=int, maxlen=max_features)
    labels = keras.utils.np_utils.to_categorical(labels, 2)
    return x1, x2, labels
 
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    parser.add_argument('--model_prefix', default='quora_model', type=str, help='Prefix for model persistance')
    args = parser.parse_args()
    return args

def start(args):
    if args.mode == 'train':
        train(args.model_prefix)
    else:
        test(args.model_prefix)

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
    while True:
        reader = pd.read_csv(source, header=0, chunksize=chunksize)
        for data in reader:
            x1, x2, y = process_data(data, tokenizer)
            #yield {'x_input': x}, {'output': y}
            yield {'x1_input': x1, 'x2_input': x2}, {'output': y}

def read_data(source, tokenizer):
    data = pd.read_csv(source, header=0)
    x1, x2, y = process_data(data, tokenizer)
    return x1, x2, y

def get_model(model_path, tokenizer=None):
    model = load_model(model_path) if isfile(model_path) else create_model(tokenizer)
    return model

def plot_history(history):
    # summarize history for accuracy
    keys = ['acc', 'loss',] #val_loss
    for k in keys:
        plt.plot(history.history[k])
    plt.legend(keys, loc='upper right')
    plt.title('model accuracy')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.show()

def train(model_prefix):
#    model_path, model_chkpt_path = model_path_tmpl.format(model_prefix), model_chkpt_path_tmpl.format(model_prefix)
    tokenizer = get_tokenizer(tokenizer_path, source=train_data_source, train=False)
    model = get_model(model_path, tokenizer)
    chkpt = ModelCheckpoint(model_chkpt_path, monitor='acc', verbose=1)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
    plotlog = PlotLog()
    tsboard = TensorBoard('../build/quora.log')
    history = model.fit_generator(generate_data(train_data_source, tokenizer), callbacks=[chkpt, early_stop, plotlog, tsboard],\
                    verbose=1, steps_per_epoch=steps_per_epoch, epochs=total_epochs, initial_epoch=init_epoch)# workers=4, pickle_safe=True)
#    history = model.fit({'x1_input':x1, 'x2_input':x2}, y, nb_epoch=total_epochs, batch_size=chunksize, verbose=1, validation_split=0.1)
    model.save(model_path)
    plot_history(history)

def do_test(model):
    res = model.predict_generator(generate_data(test_data_source), steps=1000, verbose=1)
    print('predict:', res)
    return res.argmax(1)

def test(model_prefix):
    model_path, model_chkpt_path = model_path_tmpl.format(model_prefix), model_chkpt_path_tmpl.format(model_prefix)
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

