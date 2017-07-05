#!/usr/bin/env python3
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from pandas import read_csv, DataFrame
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.text import one_hot
from os.path import isfile
import multiprocessing as mp
import argparse 


class ModelBase(object):

  modes = ['train', 'test', 'eval']
  
  @staticmethod
  def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=ModelBase.modes)
    parser.add_argument('--model_id', default='model-{}'.format(np.random.randint(0xffff)), type=str, help='Prefix for model persistance')
    parser.add_argument('--init_epoch', default=0, type=int, help='Initial epoch')
    parser.add_argument('--epochs', default=1000, type=int, help='Total epoch to run')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    args = parser.parse_args()
    return args

class Zillow(ModelBase):

  def __init__(self, args):
    self.model_id = args.model_id
    self.data_base = '../data/zillow_price'
    self.train_target_src = '{}/{}'.format(self.data_base, 'train_2016.csv')
    self.train_data_src = '{}/{}'.format(self.data_base, 'properties_2016.csv')
    self.train_preprocess = '{}/{}'.format(self.data_base, 'properties_2016_valid.csv')
    self.model_base = '../models'
    self.chkpt_path = '{}/{}.chkpt'.format(self.model_base, self.model_id)
    self.batch_size = args.batch_size
    self.feat_size = 56
    self.epochs = args.epochs
    self.lr = args.lr
    #self.pool = mp.Pool(32)

  def pre_load_once(self, src, dest, chunksize):
    """
    Columns:
    --------------
       'parcelid', 'airconditioningtypeid', 'architecturalstyletypeid',
       'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
       'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertylandusetypeid',
       'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
       'censustractandblock'
    """
    reader = read_csv(src, header=0, chunksize=chunksize, index_col=False)
    def foreach_chunk(chunk):
      cross = chunk[chunk['parcelid'].isin(self.target['parcelid'])]
      for iid in cross['parcelid']:
        ret = cross[cross['parcelid']==iid].join(self.target.set_index('parcelid')['logerror'])
        ret.to_csv(dest, index=False) if not isfile(dest) else ret.to_csv(dest, mode='a', header=False, index=False)
    yield from [foreach_chunk(chunk) for chunk in reader]

  def load_once(self, src, chunksize):
    reader = read_csv(src, header=0, chunksize=chunksize, index_col=False)
    def foreach_chunk(chunk): 
      parcelid, features, target = chunk.iloc[:,0], chunk.iloc[:,1:-2], chunk.iloc[:,-1]
      excluded = ['propertyzoningdesc']
      for e in excluded:
        features[e] = features[e].apply(lambda x: 0)

      toint = []
      for t in toint:
        features[t] = features[t].apply(lambda x: int('0x'+str(x), 16))
      # through one hot
      t = 'propertycountylandusecode'
      features[t] = features[t].fillna(0)
      features[t] = pd.Series([one_hot(str(f), 25)[0] for f in features[t]])

      tobool = ['taxdelinquencyflag']
      for t in tobool:
        features[t] = features[t].apply(lambda x: x==1 if x=='Y' else 0)

      features = features.fillna(0.0)
      #features = features.values.reshape((64, 56))
      yield parcelid.values, features, target.values
    for chunk in reader:
      yield from foreach_chunk(chunk)

  def generate_data(self, src, chunksize):
    while True:
      yield from self.load_once(src, chunksize)

  def get_valid_feat(self, nan_cnt):
    """
    valid_features: feature-name => count of rows ``feature`` is not NaN
    """
    df = DataFrame({'feat':list(nan_cnt.keys()), 'cnt':list(nan_cnt.values())})
    def cnt_lgt(level):
      print('cnt: <={}'.format(level), df[df['cnt']<=level].count())
    [cnt_lgt(l) for l in [50000, 250000, 500000, 750000, 1000000, 2500000, 5000000]]
    self.valid_feat = df[df['cnt']<=5000000]
    self.feat_size = len(self.valid_feat)

  def preprocess(self):
    self.load_target(self.train_target_src)
    self.pre_load_once(self.train_data_src, self.train_preprocess, self.batch_size)

  def load_target(self, src):
    """
    Columns:
    ---------------
    parcelid,logerror,transactiondate
    """
    self.target = read_csv(src, header=0, index_col=False)
    self.target_id = self.target.iloc[:,0]
    self.target_logerror = self.target.iloc[:,1]

  def train(self):
  
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    # forward
    self.model_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.feat_size), name='self.model_input')
    self.model_target = tf.placeholder(tf.float32, shape=(self.batch_size,), name='self.model_target')

    with tf.variable_scope('dense0') as scope:
      self.weights = tf.Variable(tf.random_normal(shape=[self.feat_size, self.batch_size], dtype=tf.float32), name='weights')
      self.bias = tf.Variable(tf.random_normal(shape=[self.batch_size,], dtype=tf.float32), name='bias')
      dense0 = tf.nn.relu(tf.add(tf.matmul(self.model_input, self.weights), self.bias))

    # LASSO
    #lasso = tf.constant(0.9)
    #self.heavy = tf.truediv(1.0, tf.add(1.0, tf.exp(tf.multiply(-50.0, tf.subtract(weights, lasso)))))
    #self.regularization_param = tf.multiply(self.heavy, 99.0)
    #self.loss = tf.add(tf.reduce_mean(tf.square(self.model_target-dense0)), self.regularization_param)
    #T.sum(T.pow(prediction-y,2))/(2*num_samples)
    self.delta = tf.subtract(dense0, self.model_target)
    self.loss = tf.reduce_sum(tf.square(self.delta))/(2*self.feat_size)
    
    # backward
    self.opt = tf.train.GradientDescentOptimizer(self.lr)
    self.grad = self.opt.minimize(self.loss,
      global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())
    losses = []
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for e in range(self.epochs):
        self.one_epoch(sess, losses)
        print('chkpt saved @ {}'.format(self.saver.save(sess, self.chkpt_path, global_step=self.global_step)))
  
  def one_epoch(self, sess, losses):
    for step, (iid, x, y) in enumerate(self.generate_data(self.train_preprocess, self.batch_size)):
      loss_val = sess.run([
          self.loss,
          #self.grad,
          ], feed_dict={
          self.model_input: x,
          self.model_target: y
          })
      print('[{}/{}] loss:{}'.format(step, tf.train.global_step(sess, self.global_step), loss_val), end='\r')
      losses.append(loss_val)

def plot_nan(nan_cnt, axs):
  ax = sns.barplot(y=list(nan_cnt.keys()), x=list(nan_cnt.values()), color='b', orient='h', ax=axs, linewidth=1.5)
  ax.set_title('NaN Count')
  fig.canvas.draw()

def init_plot():
  sns.plt.ion()
  #fig = plt.figure(figsize=(12, 8), facecolor='darkgray', edgecolor='black')
  fig, axs = sns.plt.subplots(ncols=1)
  sns.plt.show()
  return fig, axs

if __name__ == '__main__':
  #fig, axs = init_plot()
  args = Zillow.init()
  z = Zillow(args)
  #load_nan(axs)
  #z.preprocess()
  z.train()




