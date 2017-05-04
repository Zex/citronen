from apps.common import StateReport
import matplotlib
matplotlib.use("TkAgg")
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
from keras.models import load_model
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.preprocessing.text import one_hot
from keras.initializers import TruncatedNormal
import keras.backend as K
import numpy as np
from os.path import isfile
import argparse

train_data_source = "../data/russian_housing/train.csv"
eval_data_source = "../data/russian_housing/eval.csv"
test_data_source = "../data/russian_housing/test.csv"
chunksize = 10
max_features = 290   # remove id and price
model_id = 'russian_housing'
strep_log, rep_log, model_path, model_chkpt_path, tsboard_log = [None]*5
steps_per_epoch, init_epoch, total_epochs = 10, 0, 3
is_training, is_evaluating = False, False

def create_model():
    x_input = Input(shape=(max_features,), dtype='float32', name='input')
    x = Reshape((max_features, 1))(x_input)
    x = BatchNormalization()(x)
    x = Conv1D(512, 7, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True,
        kernel_initializer=TruncatedNormal(), bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.02))(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(128, kernel_initializer='uniform', activation='softmax')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128, kernel_initializer='uniform', activation='softmax')(x)
    x = Dropout(0.2)(x)

    y = Dense(1, kernel_initializer='uniform', activation='softmax', name='output')(x)

    model = Model(inputs=[x_input], outputs=[y], name='rushs')
    sgd = SGD(lr=0.01)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['acc', 'sparse_categorical_accuracy', 'binary_accuracy'])

    model.summary()
    plot_model(model, to_file='{}.png'.format(model_path), show_shapes=True, show_layer_names=True)
    print('model_path:{} steps:{} epochs:{}/{} chunksize:{} max_features:{}'.format(
           model_path, steps_per_epoch, init_epoch, total_epochs, chunksize, max_features))
    return model

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'eval'])
    parser.add_argument('--prefix', default=model_id, type=str, help='Prefix for model persistance')
    args = parser.parse_args()
    return args

def start(args):
    global is_training
    global is_evaluating
    update_path(args.prefix)

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
    keys = ['acc', 'loss',] #val_loss
    for k in keys:
        plt.plot(history.history[k])
    plt.legend(keys, loc='upper right')
    plt.title('model accuracy')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.show()

def update_path(prefix):
    global model_path
    global model_chkpt_path
    global tsboard_log
    global logfd
    global strep_log
    global rep_log
    model_id = prefix
    model_path = "../models/{}.h5".format(model_id)
    model_chkpt_path = "../models/"+model_id+"_chkpt_{epoch:02d}-{acc:.2f}.h5"
    tsboard_log = '../build/{}.log'.format(model_id)
    strep_log = '../build/log-{}.csv'.format(model_id)
    rep_log = '../build/log-{}.rep'.format(model_id)

def train():
    model = get_model(model_path)
    chkpt = ModelCheckpoint(model_chkpt_path, monitor='acc', verbose=1)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
    rep = StateReport(strep_log)
    tsboard = TensorBoard(tsboard_log)
    history = model.fit_generator(generate_data(train_data_source),
                callbacks=[chkpt, early_stop, tsboard, rep],\
                verbose=1, steps_per_epoch=steps_per_epoch, epochs=total_epochs, initial_epoch=init_epoch)
    model.save(model_path)
    plot_history(history)

def preprocess(source):
    data = read_csv(source, header=0)
    data = data.fillna(0.0)
    data.replace(('yes', 'no'), (1, 0), inplace=True)
    data.fillna(0.0)
    ids, y = data['id'], data['price_doc']
    time = data['timestamp'].astype('datetime64')
    product_type, sub_area, ecology = data['product_type'], data['sub_area'], data['ecology']
    data['product_type'] = np.reshape([one_hot(x, n=np.unique(product_type).shape[0]+1) for x in product_type], product_type.shape)
    data['sub_area'] = np.reshape([one_hot(x, n=np.unique(sub_area).shape[0]+1, split=',') for x in sub_area], sub_area.shape)
    data['ecology'] = np.reshape([one_hot(x, n=np.unique(ecology).shape[0]+1) for x in ecology], ecology.shape)
    intermedia = '../build/{}-preprocess.csv'.format(model_id)
    data.to_csv(intermedia)
    #x = data.iloc[0:chunksize, 13:30]#, 12:291]
    #return x, y

def generate_data(source): 
    while True:
        reader = read_csv(source, header=0, chunksize=chunksize)
        for chunk in reader:
            x, y = preprocess(chunk)
            yield x, y

if __name__ == '__main__':
    args = init()
#    start(args)
    #for x, y in generate_data(train_data_source):
        #print(x, y)
    #    break
    preprocess(train_data_source)


