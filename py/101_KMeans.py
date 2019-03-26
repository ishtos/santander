#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 2018

@author: toshiki.ishikawa
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import utils
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from time import time, sleep
from itertools import combinations
from multiprocessing import cpu_count, Pool
from sklearn.cluster import KMeans

utils.start(__file__)

#==============================================================================
# SETTINGS
#==============================================================================
PATH = os.path.join('..', 'data')

PREFIX = 'ss'

POSTFIX = '101'

#==============================================================================
# READ
#==============================================================================
train = pd.read_csv(os.path.join(PATH, f'{PREFIX}_train.csv'))
test = pd.read_csv(os.path.join(PATH, f'{PREFIX}_test.csv'))

var_columns = [f'var_{i}' for i in range(0, 200)]
#==============================================================================
# CONCAT
#==============================================================================
df = pd.concat([train, test], axis=0)

#==============================================================================
# K MEANS
#==============================================================================
for i in tqdm(range(2, 30)):
    km = KMeans(n_clusters=i, n_init=30, n_jobs=-1, random_state=0)
    df[f'kmeans_{i}'] = km.fit_predict(df[var_columns])
#     distance = km.fit_transform(df[var_columns])
#     for j in range(0, i):
#         df[f'kmeans_{i}_{j+1}_distance'] = distance[:, j]  

for i in tqdm([30, 50, 100, 200]):
    km = KMeans(n_clusters=i, n_init=200, n_jobs=-1, random_state=0)
    df[f'kmeans_{i}'] = km.fit_predict(df[var_columns])
#     distance = km.fit_transform(df[var_columns])
#     for j in range(0, i):
#         df[f'kmeans_{i}_{j+1}_distance'] = distance[:, j] 

#==============================================================================
# TO CSV
#==============================================================================
var_columns += ['target']
use_columns = [f for f in df.columns if f not in var_columns]

df[use_columns].to_csv(os.path.join('..', 'feature', f'{PREFIX}_{POSTFIX}.csv'), index=False)
#==============================================================================

utils.end(__file__)
