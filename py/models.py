import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedKFold

def cv_lightgbm(X, y, X_test, NFOLDS=5, SEED=6):
    params_in_train = {
        'num_boost_round': 20000,
        'early_stopping_rounds': 200,
        'verbose_eval': 500,
    }

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

    return oof, predictions, scores, feature_importance_df
