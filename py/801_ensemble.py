#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019
@author: toshiki.ishikawa
"""

import os
import sys
import gc
import utils

import numpy as np
import pandas as pd


utils.start(__file__)
# =============================================================================

PATH = os.path.join('..', 'submission')

# =============================================================================
# ENSEMBLE
# =============================================================================

lgb = pd.read_csv(os.path.join(PATH, 'baseline_0305.csv'))
xl = pd.read_csv(os.path.join(PATH, 'xlearn.csv'))

lgb['target'] = lgb['target'] * 0.95 + xl['target'] * 0.05

lgb.to_csv(os.path.join(PATH, 'ensemble.csv'), index=False)

# =============================================================================
utils.end(__file__)
