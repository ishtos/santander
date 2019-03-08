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


utils.start(__file__)

#==============================================================================

PATH = os.path.join('..', 'input')

train = pd.read_csv(os.path.join(PATH, '/train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

#==============================================================================
# df
#==============================================================================

train.to_csv(os.path.join('..', 'data', 'train.csv'), index=False)
test.to_csv(os.path.join('..', 'data', 'test.csv'), index=False)
#==============================================================================

utils.end(__file__)
