#!/usr/bin/env ptargetthon3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019

@author: toshiki.ishikawa
"""
import os
import gc 
import sys
import datetime
import utils

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xlearn as xl

from tqdm import tqdm
from multiprocessing import cpu_count

utils.start(__file__)
# =============================================================================
DATA_PATH = os.path.join('..', 'data')
INPUTS_PATH = os.path.join('..', 'input')
OUTPUTS_PATH = os.path.join('..', 'output')
SUBMISSION_PATH = os.path.join('..', 'submission')

NFOLDS = 11

param = {
    'task': 'binary',
    'metric': 'auc', 
    'opt': 'adagrad',
    
    'lr': 0.1, 
    'lambda': 0.002, 
    'epoch': 20,
}

# =============================================================================
# MODEL 
# =============================================================================
for i in range(NFOLDS):
    ffm_model = xl.create_ffm()
    ffm_model.setTrain(os.path.join(DATA_PATH, f'train_fold{i}.txt'))
    ffm_model.setValidate(os.path.join(DATA_PATH, f'valid_fold{i}.txt'))

    ffm_model.fit(param, os.path.join(OUTPUTS_PATH, f'model_fold{i}.out'))

    ffm_model.setTest(os.path.join(DATA_PATH, 'test.txt'))
    ffm_model.setSigmoid() 
    ffm_model.predict(os.path.join(OUTPUTS_PATH, f'model_fold{i}.out'), os.path.join(OUTPUTS_PATH, f'prediction_fold{i}.txt'))

# =============================================================================
# SUBMISSION
# =============================================================================
submission = pd.read_csv(os.path.join(INPUTS_PATH, 'sample_submission.csv'))
for i in range(NFOLDS):
    pred = pd.read_csv(os.path.join(OUTPUTS_PATH, f'prediction_fold{i}.txt'), header=None).rename(columns={0: 'target'}) 
    submission['target'] = pred['target']

submission['target'] = submission['target'] / NFOLDS
submission.to_csv(os.path.join(SUBMISSION_PATH, 'xlearn.csv'), index=False)

# =============================================================================
utils.end(__file__)
