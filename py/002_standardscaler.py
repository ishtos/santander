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
from sklearn.preprocessing import StandardScaler

utils.start(__file__)

#==============================================================================
# SETTINGS
#==============================================================================
PATH = os.path.join('..', 'data')

train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

test.insert(1, 'target', -1)

var_columns = [f'var_{i}' for i in range(0, 200)]
#==============================================================================
# CONCAT
#==============================================================================
df = pd.concat([train, test], axis=0)

#==============================================================================
# STANDARD SCALER
#==============================================================================
ss = StandardScaler()
df[var_columns] = ss.fit_transform(df[var_columns])

#==============================================================================
# TO CSV
#==============================================================================
train = df.query('target != -1')
test = df.query('target == -1')

train.to_csv(os.path.join('..', 'data', 'ss_train.csv'), index=False)
test.to_csv(os.path.join('..', 'data', 'ss_test.csv'), index=False)
#==============================================================================

utils.end(__file__)
