#!/usr/bin/env python3
import matplotlib
matplotlib.use("TkAgg")
from model_base import ModelBase
from pandas import read_csv, DataFrame
from keras.preprocessing.text import one_hot
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import xgboost as xgb
from os.path import isfile
import multiprocessing as mp
import pandas as pd
import seaborn as sns
import numpy as np


class ZillowBase(ModelBase):

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
    target_reader = read_csv(self.train_target_preprocess, header=0, chunksize=chunksize, index_col=False)

    def feature_process(features):
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
      return features

    def foreach_chunk(chunk): 
      parcelid, features, target = chunk.iloc[:,0], chunk.iloc[:,1:-2], chunk.iloc[:,-1]
      yield parcelid.values, feature_process(features), target.values
    
    for chunk, target in zip(reader, target_reader):
      #yield from foreach_chunk(chunk)
      yield chunk.iloc[:,0].values, feature_process(chunk.iloc[:,1:]).values, target.values

  def generate_data(self, src, chunksize, forever=False):
    while True:
      yield from self.load_once(src, chunksize)
      if not forever:
        break

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
    #self.target_id = self.target.iloc[:,0]
    #self.target_logerror = self.target.iloc[:,1]

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


class Zillow(ZillowBase):

  def train(self):
    lr = LinearRegression() 
    for step, (iid, x, y) in enumerate(self.generate_data(self.train_preprocess, self.batch_size)):
      lr.fit(x, y)
      pred = lr.predict(x)
      print('[{}] P ===> [{}]'.format(step, pred), flush=True)
      score = pred = lr.score(x, y)
      print('[{}] S ===> [{}]'.format(step, score), flush=True)


if __name__ == '__main__':
  #fig, axs = init_plot()
  args = Zillow.init()
  z = Zillow(args)
  #z.preprocess()
  z.train()




