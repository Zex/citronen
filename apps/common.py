from keras.callbacks import Callback

class StateReport(Callback):
    def __init__(self, output_path):
        self.w = open(output_path, 'w+')

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.w.write('epoch:{} '.format(epoch)+''.join(['{}:{} '.format(k, v) for k,v in logs.items()])+'\n')
        #self.w.write('{},{},{},{}\n'.format(logs.get('loss'), logs.get('acc'), logs.get('binary_accuracy'), logs.get('sparse_categorical_accuracy')))

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.w.write('batch:{} '.format(batch)+''.join(['{}:{} '.format(k, v) for k,v in logs.items()])+'\n')

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        self.w.write(''.join(['{}:{} '.format(k, v) for k,v in logs.items()])+'\n')


