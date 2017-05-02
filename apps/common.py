from keras.callbacks import Callback

class PlotLog(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        PlotLog.logfd.write('{},{},{},{}\n'.format(logs.get('loss'), logs.get('acc'), logs.get('binary_accuracy'), logs.get('sparse_categorical_accuracy')))
        PlotLog.logfd.flush()

