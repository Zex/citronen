from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Embedding
from keras.models import load_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.optimizers import SGD
#from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
#import keras.backend as K
import keras
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import string
from os.path import isfile
import numpy as np

shrink_data_source = '../data/quora/train_shrink.csv'
train_data_source = '../data/quora/origin-train.csv'
validation_data_source = '../data/quora/validation.csv'
test_data_source = '../data/quora/test.csv'
test_result = '../data/quora/test_result.csv'
#model_path = "../models/quora_mlp.pkl"
model_path = "../models/quora_model.h5"
model_path_tmpl = "../models/{}.h5"
model_chkpt_path_tmpl = "../models/{name:s}_chkpt_{epoch:02d}-{acc:.2f}.h5"
model_chkpt_path = "../models/quora_model_chkpt_{epoch:02d}-{acc:.2f}.h5"
max_features = 32
max_encoded_len = 27
chunksize = 128
learning_rate = 0.02
max_epochs = 1000
token = Tokenizer()

def preprocess(q):
    q = [x.lower() for x in q]
    q = [''.join(c for c in x if c not in string.punctuation) for x in q]
    q = [''.join(c for c in x if c not in '0123456789') for x in q]
    q = [' '.join(x.split()) for x in q]
    return q

def create_model():
    ## with tfidf input
    x1_input = Input(shape=(max_features*2,), dtype='int32', name='x_input')
#    x2_input = Input(shape=(max_features,), dtype='int32', name='x2_input')

    x = Embedding(output_dim=max_features*2, input_dim=100000, input_length=max_features*2)(x1_input)
    x = Conv1D(max_features*2, 5, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(max_features*2, 5, activation='relu')(x)
    x = MaxPooling1D(1)(x)
    x = Conv1D(max_features*2, 8, activation='relu')(x)
    x = MaxPooling1D(8)(x)
    x = Conv1D(max_features*2, 2, activation='relu')(x)
    x = MaxPooling1D(1)(x)
#    x = concatenate([x1, x2])
#    x = LSTM(128, return_sequences=False, name='x_lstm1', dropout=0.2)(x)
#    x = LSTM(64, return_sequences=True, name='x_lstm2', dropout=0.2)(x)
#    x = LSTM(32, return_sequences=False, name='x_lstm3', dropout=0.2)(x1)
    x = Flatten()(x)
    x = Dense(500, kernel_initializer='uniform', activation='relu', name='x_dense1')(x)
    x = Dropout(0.2)(x)
    x = Dense(200, kernel_initializer='uniform', activation='relu', name='x_dense2')(x)
    x = Dropout(0.2)(x)
    x = Dense(100, kernel_initializer='uniform', activation='relu', name='x_dense3')(x)
    x = Dropout(0.2)(x)
    y = Dense(2, kernel_initializer='uniform', activation='softmax', name='output')(x)
    model = Model(inputs=[x1_input], outputs=[y],  name='final')
#    model = Model(inputs=[x1_input, x2_input], outputs=[y],  name='final')
#    rmsprop = RMSprop(lr=learning_rate)
    sdg = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['acc'])

    model.summary()
    plot_model(model, to_file='{}.png'.format(model_path), show_shapes=True, show_layer_names=True)
    return model

def process_data(data):
    # process train data
    q1, q2, labels = data['question1'].values.astype('U'),\
                     data['question2'].values.astype('U'),\
                     data['is_duplicate'].values.astype(np.int32)
    token.fit_on_texts(q1)
    token.fit_on_texts(q2)
    x1 = token.texts_to_sequences(q1)  
    x2 = token.texts_to_sequences(q2)
    #cosine_similarity = np.dot(x1, x2.T)/(np.linalg.norm(x1)*np.linalg.norm(x2))
    #print(cosine_similarity)
    x1 = pad_sequences(x1, padding='post', truncating='post', dtype=int, maxlen=max_features)
    x2 = pad_sequences(x2, padding='post', truncating='post', dtype=int, maxlen=max_features)
    x = np.concatenate((np.asarray(x1), np.asarray(x2)), axis=1)
    labels = keras.utils.np_utils.to_categorical(labels, 2)
    return x, labels
  
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
    parser.add_argument('--model_prefix', default='quora_model', type=str, help='Prefix for model persistance')
    args = parser.parse_args()
    return args

def start(args):
    if args.mode == 'train':
        train(args.model_prefix)
    else:
        test(args.model_prefix)

def generate_data(source):
    while True:
        reader = pd.read_csv(source, header=0, chunksize=chunksize)
        for data in reader:
            x, y = process_data(data)
            yield {'x_input': x}, {'output': y}
            #yield {'x1_input': x1, 'x2_input': x2}, {'output': y}

def get_model(model_path):
    model = None
    if isfile(model_path):
        model = load_model(model_path)
    else:
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

def train(model_prefix):
#    model_path, model_chkpt_path = model_path_tmpl.format(model_prefix), model_chkpt_path_tmpl.format(model_prefix)
#    print(model_path, model_chkpt_path)
    model = get_model(model_path)
    chkpt = ModelCheckpoint(model_chkpt_path, monitor='acc', verbose=1)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
    history = model.fit_generator(generate_data(train_data_source), callbacks=[chkpt, early_stop],\
                    verbose=1, steps_per_epoch=3700, epochs=20, initial_epoch=0)# workers=4, pickle_safe=True)
    model.save(model_path)
    plot_history(history)

def do_test(model):
#   steps = round(lines/chunksize)
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

