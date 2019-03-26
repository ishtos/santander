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
from sklearn.neighbors import KNeighborsClassifier

utils.start(__file__)

#==============================================================================
# SETTINGS
#==============================================================================
PATH = os.path.join('..', 'data')

PREFIX = 'ss'

POSTFIX = '108'

#==============================================================================
# READ
#==============================================================================
train = pd.read_csv(os.path.join(PATH, f'{PREFIX}_train.csv'))
test = pd.read_csv(os.path.join(PATH, f'{PREFIX}_test.csv'))

var_columns = [f'var_{i}' for i in range(0, 200)]

#==============================================================================
# KNEIGHBORSCLASSIFIER
#==============================================================================
knc = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knc.fit(train[var_columns], train['target'])

train_proba = knc.predict_proba(train[var_columns])
train['knc_0'] = train_proba[:, 0]
train['knc_1'] = train_proba[:, 1]

test_proba = knc.predict_proba(test[var_columns])
test['knc_0'] = test_proba[:, 0]
test['knc_1'] = test_proba[:, 1]

#==============================================================================
# TO CSV
#==============================================================================
df = pd.concat([train, test], axis=0)

var_columns += ['target']
use_columns = [f for f in df.columns if f not in var_columns]

df[use_columns].to_csv(os.path.join('..', 'feature', f'{PREFIX}_{POSTFIX}.csv'), index=False)
#==============================================================================

utils.end(__file__)
