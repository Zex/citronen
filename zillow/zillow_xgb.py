#!/usr/bin/env python3
import matplotlib
matplotlib.use("TkAgg")
from model_base import ModelBase
import numpy as np
from pandas import read_csv, DataFrame
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
import xgboost as xgb
from keras.preprocessing.text import one_hot
from os.path import isfile
import multiprocessing as mp
import argparse 
import sys


class Zillow(ModelBase):

  def __init__(self, args):
    self.model_id = args.model_id
    self.data_base = '../data/zillow_price'
    self.train_target_src = '{}/{}'.format(self.data_base, 'train_2016.csv')
    self.train_data_src = '{}/{}'.format(self.data_base, 'properties_2016.csv')
    self.train_preprocess = '{}/{}'.format(self.data_base, 'properties_2016_valid.csv')
    self.train_target_preprocess = '{}/{}'.format(self.data_base, 'train_target_valid.csv')
    self.model_base = '../models'
    self.model_path = "{}/{}.xgb".format(self.model_base, self.model_id)
    self.chkpt_path = '{}/{}.chkpt'.format(self.model_base, self.model_id)
    self.batch_size = args.batch_size
    self.feat_size = 56
    self.epochs = args.epochs
    self.lr = args.lr
    #self.pool = mp.Pool(32)

  def pre_load_once(self, src, chunksize):
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
    def foreach_chunk(chunk):
      def persis(dest, df):
        if not isfile(dest):
          df.to_csv(dest, index=False)
        else:
          df.to_csv(dest, mode='a', header=False, index=False)
      cross = chunk[chunk['parcelid'].isin(self.target['parcelid'])]
      feats = cross.keys()
      for iid in cross['parcelid']:
        data = cross[cross['parcelid']==iid]
        if data.iloc[0,1:].isnull().all():
          continue
        lbl = pd.DataFrame(self.target[self.target['parcelid']==iid]['logerror'])
        if len(data) == 0 or len(lbl) == 0:
          continue
        persis(self.train_preprocess, data)
        persis(self.train_target_preprocess, lbl)
    reader = read_csv(src, header=0, chunksize=chunksize, index_col=False)
    [foreach_chunk(chunk) for chunk in reader]

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
      t = 'longitude'
      features[t] = features[t].apply(lambda x: 0-x if x<0 else x)
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
    self.pre_load_once(self.train_data_src, self.batch_size)

  def load_target(self, src):
    """
    Columns:
    ---------------
    parcelid,logerror,transactiondate
    """
    self.target = read_csv(src, header=0, index_col=False)
    print(self.target.shape)
    #self.target_id = self.target.iloc[:,0]
    #self.target_logerror = self.target.iloc[:,1]

  def train(self):
    #for step, (iid, x, y) in enumerate(self.generate_data(self.train_preprocess, self.batch_size)):
    #  print(type(x), type(y))
    if True:
      self.load_target(self.train_target_preprocess)
      dm = xgb.DMatrix(self.train_preprocess, label=self.target, missing=0.0)
      params = {
        'objective': 'reg:linear',
        'rate_drop': 0.3,
        #'max_depth': 10,
        #'eta': 10,
        #'silent': 1
      }
      model_path = self.model_path if isfile(self.model_path) else None
      try:
        self.bst = xgb.train(params, dm, xgb_model=model_path)#, learning_rates=self.lr)
      except Exception as ex:
        print('xgb train failed', ex)
        sys.exit()
      self.bst.save_model(self.model_path)
      pred = self.bst.predict(dm)
      print('predict:{}'.format(pred))

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


def with_torch():

  import torch.nn as nn

  x = torch.randn(32,32)
  y = torch.randn(32)
  lstm1, lstm2 = nn.LSTMCell(1, 64), nn.LSTMCell(64, 1)



if __name__ == '__main__':
  #fig, axs = init_plot()
  args = Zillow.init()
  z = Zillow(args)
  #z.preprocess()
  z.train()




