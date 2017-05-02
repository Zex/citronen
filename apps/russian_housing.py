from apps.common import PlotLog
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
import keras.backend as K
from os.path import isfile
import argparse

train_data_source = "../data/russian_housing/train.csv"
eval_data_source = "../data/russian_housing/eval.csv"
test_data_source = "../data/russian_housing/test.csv"
chunksize = 32
max_features = 290   # remove id and price
model_id = 'russian_housing'
model_path, model_chkpt_path, tsboard_log = [None]*3
steps_per_epoch, init_epoch, total_epochs = 1000, 0, 1000
is_training, is_evaluating = False, False

def create_model():
    x_input = Input(shape=(max_features,), dtype='float32', name='input')
    x = BatchNormalization()(x_input)
    x = Reshape((max_features, 1))(x)
    x = Conv1D(512, 7, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.02))(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(512, 7, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.02))(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    x = Conv1D(512, 7, strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=regularizers.l2(0.01),
        bias_regularizer=regularizers.l2(0.01),
        activity_regularizer=regularizers.l1(0.02))(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid')(x)
    
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(128, kernel_initializer='uniform', activation='softmax')(x)
    x = Dropout(0.2)(x)
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
    model_id = prefix
    model_path = "../models/{}.h5".format(model_id)
    model_chkpt_path = "../models/"+model_id+"_chkpt_{epoch:02d}-{acc:.2f}.h5"
    tsboard_log = '../build/{}.log'.format(model_id)

def train():
    model = get_model(model_path)
    chkpt = ModelCheckpoint(model_chkpt_path, monitor='acc', verbose=1)
    early_stop = EarlyStopping(monitor='loss', verbose=1, patience=3, min_delta=0.0001)
    plotlog = PlotLog()
    tsboard = TensorBoard(tsboard_log)
    PlotLog.logfd = open('../build/log-{}.csv'.format(model_id), 'w+')
    history = model.fit_generator(generate_data(train_data_source),
                callbacks=[chkpt, early_stop, plotlog, tsboard],\
                verbose=1, steps_per_epoch=steps_per_epoch, epochs=total_epochs, initial_epoch=init_epoch)
    model.save(model_path)
    plot_history(history)

def generate_data(source): 
    while True:
        reader = read_csv(source, header=0, chunksize=chunksize)
        for chunk in reader:
            ids, y = chunk['id'], chunk['price_doc']

if __name__ == '__main__':
    args = init()
    start(args)


