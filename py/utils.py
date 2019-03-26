#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 4 2019
@author: toshiki.ishikawa
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp

from glob import glob
from tqdm import tqdm
from time import time, sleep
from datetime import datetime
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import KFold
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from functools import reduce, partial
from scipy.stats import skew, kurtosis, iqr


# =============================================================================
# global variables
# =============================================================================
COMPETHITION_NAME = 'Santander'

# =============================================================================
# def
# =============================================================================
def start(file_name):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format(file_name, os.getpid(), datetime.today()))
    return


def reset_time():
    global st_time
    st_time = time()
    return


def end(file_name):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(file_name))
    print('time: {:.2f}min'.format(elapsed_minute()))
    return


def elapsed_minute():
    return (time() - st_time)/60


def reduce_mem_usage(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # if c_min > np.info(np.int8).min and c_max < np.iinfo(np.int8).max:
                #     df[col] = df[col].astype(np.int8)
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

