#
#
import os
import sys
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt



def preprocess():
    path = 'data/price/train.tsv'
    df = pd.read_csv(path, delimiter='\t')
    df = df.dropna()
    df = df.drop_duplicates()
    
    grp = df.groupby('category_name') 
    print(dir(grp))
    print(grp.category_name)



if __name__ == '__main__':
    preprocess()
