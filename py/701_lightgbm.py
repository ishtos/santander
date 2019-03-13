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
import lightgbm as lgb 

from tqdm import tqdm
from collections import defaultdict
from multiprocessing import cpu_count

from sklearn.model_selection import StratifiedKFold, KFold 
from sklearn.metrics import roc_auc_score

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

NTHREAD = cpu_count()

NFOLDS = 5

# =============================================================================
# READ DATA
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

# =============================================================================
# ADD HANDCRAFTED FEATURES
# =============================================================================

# for column in train.columns[2:]:
#     train[column] = 1 / (train[column] + 1)
#     test[column] = 1 / (test[column] + 1)

# for column in train.columns[2:]:
#     train[column] = np.log1p(train[column])
#     test[column] = np.log1p(test[column])

df = pd.concat([train, test], axis=0)
for c in df.columns[2:]:
    df[c] = df[c] / (df[c].max() - df[c].min())

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]
del df
gc.collect()
# =============================================================================
# PREPROCESS
# =============================================================================
y = train['target']

not_use_cols = ['ID_code', 'target'] 
use_cols = [c for c in train.columns if c not in not_use_cols]

X = train[use_cols]
X_test = test[use_cols]

# =============================================================================
# PARAMS
# =============================================================================
params_in_train = {
    'num_boost_round': 20000,
    'early_stopping_rounds': 200,
    'verbose_eval': 500,
}

# =============================================================================
# CV
# =============================================================================
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
predictions = np.zeros(len(X_test))
scores = {'train': [], 'valid': []}
features = X.columns
feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])

for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print('fold: {}/{}'.format(fold+1, skf.n_splits))

    params = {
        'boosting': 'gbdt',
        'metric': 'auc',
        'objective': 'binary',
        'max_depth': 6,
        'num_leaves': 12,
        'min_data_in_leaf': 64,
        'bagging_freq': 5,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.4,
        'reg_alpha': 2,
        'reg_lambda': 5,
        'min_gain_to_split': 0.01,
        'min_child_weight': 19,
        'num_threads': cpu_count(),
        'verbose': -1,
        'seed': int(2**fold),
        'bagging_seed': int(2**fold),
        'drop_seed': int(2**fold),
    }
    
    dtrain = lgb.Dataset(X.iloc[train_index], label=y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[valid_index], label=y.iloc[valid_index])
    
    model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], **params_in_train)
    scores['train'].append(model.best_score['training']['auc'])
    scores['valid'].append(model.best_score['valid_1']['auc'])
    oof[valid_index] = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)

    fold_feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])
    fold_feature_importance_df['feature'] = features
    fold_feature_importance_df['importance'] = model.feature_importance()
    fold_feature_importance_df['fold'] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_feature_importance_df], axis=0)

    predictions += model.predict(X_test, num_iteration=model.best_iteration) / NFOLDS

    del model

cv_score = roc_auc_score(y, oof)**0.5
print('Shape: {}'.format(X.shape))
print('Num folds: {}'.format(NFOLDS))
print('Train Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
    np.mean(scores['train']), np.max(scores['train']), np.min(scores['train']), np.std(scores['train'])))
print('Valid Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
    np.mean(scores['valid']), np.max(scores['valid']), np.min(scores['valid']), np.std(scores['valid'])))
print('CV Score: {:<8.5f}'.format(cv_score))

logger.info('''
# ============================================================================= 
# SUMMARY                                                     
# =============================================================================
''')
logger.info('Shape: {}'.format(X.shape))
logger.info('Num folds: {}'.format(NFOLDS))
logger.info('Train Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
    np.mean(scores['train']), np.max(scores['train']), np.min(scores['train']), np.std(scores['train'])))
logger.info('Valid Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
    np.mean(scores['valid']), np.max(scores['valid']), np.min(scores['valid']), np.std(scores['valid'])))
logger.info('CV Score: {:<8.5f}'.format(cv_score))
logger.info('''
# ============================================================================= 
# END                                              
# =============================================================================
''')

submission = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))
submission['target'] = predictions
submission.to_csv(os.path.join('..', 'submission', '{}_lightgbm.csv'.format(str(datetime.datetime.today().date()).replace('-', ''))), index=False)

feature_importance_df['importance'] = feature_importance_df['importance'].astype('int')
# mean = feature_importance_df['importance'].mean()
# std = feature_importance_df['importance'].std()
# width = 24
ordered_feature = feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).index
plt.figure(figsize=(12, 48))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plot = sns.barplot(x='importance', y='feature', data=feature_importance_df, order=ordered_feature)
# plt.vlines(mean, -width, width, colors='red')
# plt.vlines(mean+std, -width, width, colors='red', linestyles=':')
fig = plot.get_figure()
fig.savefig(os.path.join('.', 'IMP_png', '{}_IMP.png'.format(str(datetime.datetime.today().date()).replace('-', ''))), bbox_inches='tight')
#==============================================================================
utils.end(__file__)