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
from sklearn.decomposition import PCA

utils.start(__file__)

#==============================================================================
# SETTINGS
#==============================================================================
PATH = os.path.join('..', 'data')

PREFIX = 'ss'

POSTFIX = '201'

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
# PCA
#==============================================================================
model = PCA(n_components=2)

result = model.fit_transform(df[var_columns])
df['pca_2_0'] = result[:, 0]
df['pca_2_1'] = result[:, 1]

#==============================================================================
# TO CSV
#==============================================================================
var_columns += ['target']
use_columns = [f for f in df.columns if f not in var_columns]

df[use_columns].to_csv(os.path.join('..', 'feature', f'{PREFIX}_{POSTFIX}.csv'), index=False)
#==============================================================================

utils.end(__file__)
