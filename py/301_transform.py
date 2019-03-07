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
import xlearn as xl

from tqdm import tqdm

utils.start(__file__)
# =============================================================================
PATH = os.path.join('..', 'input')

# =============================================================================
# DEF
# =============================================================================
def to_libffm_format(train, test, v):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    noofrows = train.shape[0]
    with open(f"../data/alltrainffm_{v}.txt", "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            datastring = ""
            datarow = train.iloc[r].to_dict()
            datastring += str(int(datarow['y']))

            for i, x in enumerate(catdict.keys()):
                if(catdict[x] == 0):
                    datastring = datastring + " " + \
                        str(i)+":" + str(i)+":" + str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":" + str(int(code))+":1"
            datastring += '\n'
            text_file.write(datastring)

    noofrows = test.shape[0]
    with open(f"../data/alltestffm_{v}.txt", "w") as text_file:
        for n, r in enumerate(range(noofrows)): 
            datastring = ""
            datarow = test.iloc[r].to_dict()
            datastring += str(int(datarow['y']))

            for i, x in enumerate(catdict.keys()):
                if(catdict[x] == 0):
                    datastring = datastring + " " + \
                        str(i)+":" + str(i)+":" + str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":" + str(int(code))+":1"
            datastring += '\n'
            text_file.write(datastring)


# =============================================================================
# MAIN
# =============================================================================

train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

categories = [c for c in train.columns if c != 'target']
numerics = []

test['y'] = 0
for v in tqdm(train['y'].unique()):
    if (v in y) and (v not in y_):
        train['y'] = train['y'].apply(lambda x: np.where(x == v, 1, 0)) 
        to_libffm_format(train, test, v)

# =============================================================================
utils.end(__file__)
