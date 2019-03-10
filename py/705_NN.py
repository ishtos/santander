#!/usr/bin/env ptargetthon3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019

@author: toshiki.ishikawa
"""
import warnings
warnings.simplefilter('ignore')

import os
import gc 
import sys
import datetime
import utils

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import tocrh 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional import F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

utils.start(__file__)
#==============================================================================
# LOGGER
#==============================================================================
from logging import getLogger, FileHandler, Formatter, DEBUG
logger = getLogger(__name__)
logger.setLevel(DEBUG)

file_handler = FileHandler(os.path.join('logs', 'log_{}'.format(str(datetime.datetime.today().date()).replace('-', ''))))
formatter = Formatter('%(message)s')
file_handler.setFormatter(formatter)
file_handler.setLevel(DEBUG)

logger.addHandler(file_handler)
logger.propagate = False

#==============================================================================
PATH = os.path.join('..', 'data')

KEY = 'ID_code'

SEED = 6
# SEED = np.random.randint(9999)

# =============================================================================
# READ DATA
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

# =============================================================================
# ADD HANDCRAFTED FEATURES
# =============================================================================

# =============================================================================
# PREPROCESS
# =============================================================================
y = train['target']

not_use_cols = ['ID_code', 'target'] 
use_cols = [c for c in train.columns if c not in not_use_cols]

X = train[use_cols]
X_test = test[use_cols]

# =============================================================================
# STANDARD SCALER
# =============================================================================
sc = StandardScaler()
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

# =============================================================================
# MODEL
# =============================================================================

class DNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=2):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

# =============================================================================
# TRAIN 
# =============================================================================
model = DNN(input_size, hidden_size, num_classes).to(device)