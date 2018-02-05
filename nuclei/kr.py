#
#
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from nuclei.provider import *
import numpy as np

fig_outpath = 'data/nuclei/pred.png'

def iou_loss(labels, predictions):
    inter = labels*predictions
    inter = inter/labels
    inter = 1.0-tf.reduce_mean(inter, name='loss')
    return inter


class KR(object):

    def __init__(self, args):
        super(KR, self).__init__()
        self.epochs = args.epochs
        self.lr = args.lr
        self.init_step = args.init_step
        self.dropout_rate = args.dropout_rate
        self.summ_intv = args.summ_intv
        self.model_dir = args.model_dir
        self.batch_size = args.batch_size
        self.log_path = os.path.join(self.model_dir, 'cnn')
        self.relu_alpha = 0.01

        self.prov = Provider()
        self.height, self.width = self.prov.height, self.prov.width
        self.channel = self.prov.channel
        self.device = "/cpu:0"


    def _build_model(self):
        input_x = Input(shape=(self.height, self.width, self.channel), dtype='float32', name='input_x')
        input_y = Input(shape=(self.height, self.width, 1), dtype='float32', name='input_y')
        
        x = Conv2D(5, [3, 3], strides=(1, 1), activity_regularizer=LeakyReLU(self.relu_alpha), kernel_initializer='random_uniform')(input_x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = Conv2D(5, [3, 3], strides=(1, 1), activity_regularizer=LeakyReLU(self.relu_alpha), kernel_initializer='random_uniform')(input_x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)
        x = Conv2D(5, [3, 3], strides=(1, 1), activity_regularizer=LeakyReLU(self.relu_alpha), kernel_initializer='random_uniform')(input_x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(x)

        x = UpSampling2D((2, 2))(x)
#        x = UpSampling2D((3, 3))(x)
#        x = UpSampling2D((3, 3))(x)

        output = x

        self.model = Model(inputs=[input_x], outputs=[output], name='model')

        self.model.summary()
        self.model.compile(\
                loss=iou_loss,
                #sample_weight_mode='temporal',
                target_tensor=input_y,
                optimizer=Adam(lr=self.lr)
                )
        
    def train(self):
        global sess

        self._build_model()
        sess = tf.Session()
        callbacks = [
            TensorBoard(log_dir=self.log_path,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=True,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None)
            ]
        self.model.fit_generator(self.prov.gen_data(), callbacks=callbacks, steps_per_epoch=self.epochs//self.batch_size+1)
        self.model.save(self.model_path)


def start():
    obj = KR(init())
    obj.train()


if __name__ == "__main__":
    start()
