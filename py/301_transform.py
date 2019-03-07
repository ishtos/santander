#!/usr/bin/env ptargetthon3
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
from sklearn.model_selection import StratifiedKFold, KFold

utils.start(__file__)
# =============================================================================
PATH = os.path.join('..', 'input')

NFOLDS = 6

SEED = 6

# =============================================================================
# MAIN
# =============================================================================

train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))
test.insert(1, 'target', 0)

for c in train.columns[2:]:
    train[c] = np.round(train[c], 0)
    test[c] = np.round(test[c], 0)

features = train.columns[2:]

categories = []
numerics = [c for c in train.columns if c != 'target' and c != 'ID_code']

currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1

y = train['target']
kf = KFold(NFOLDS, shuffle=True, random_state=SEED)

for fold, (train_index, valid_index) in enumerate(kf.split(train, y)):
    print('fold: {}/{}'.format(fold+1, kf.n_splits))

    train_ = train.iloc[train_index]
    valid = train.iloc[valid_index]

    # =========================================================================
    # TRAIN
    # =========================================================================

    noofrows = train_.shape[0]
    noofcolumns = len(features)

    with open(f"../data/train_fold{fold}.txt", "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            datastring = ""
            datarow = train_.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

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

    # =========================================================================
    # VALID
    # =========================================================================

    noofrows = valid.shape[0]
    noofcolumns = len(features)

    with open(f"../data/valid_fold{fold}.txt", "w") as text_file:
        for n, r in enumerate(range(noofrows)):
            datastring = ""
            datarow = valid.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

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
# TEST
# =============================================================================

noofrows = test.shape[0]
noofcolumns = len(features)

with open("../data/test.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)): 
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))

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

with open("../data/test.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)): 
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))

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
utils.end(__file__)
