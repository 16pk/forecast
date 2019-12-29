# -*- coding: utf-8 -*-
import warnings
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization


def get_best_params_by_bo(x_train, y_train, x_val, y_val):
    """在给定的参数范围内使用贝叶斯方案寻找最优参数"""

    def _xgb_model(**kwargs):
        xgb_params = bo_result_to_xgb(kwargs)
        clf = xgb.XGBRegressor(booster='gbtree', n_estimators=300, verbosity=0, n_jobs=16, seed=42,
                               reg_alpha=0.1, reg_lambda=0.1, **xgb_params)
        clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse',
                verbose=False)
        eval_result = clf.evals_result()
        dev_rmse = eval_result['validation_0']['rmse'][-1]
        return -dev_rmse

    search_space = {
        'learning_rate': (0.02, 0.06),
        'max_depth': (3, 7),
        'log_gamma': (-3, 1),
        'min_child_weight': (0, 20),
        'max_delta_step': (0, 10),
        'subsample': (0.3, 0.9),
        'colsample_bytree': (0.3, 0.9)
    }

    xgb_bayes = BayesianOptimization(_xgb_model, search_space)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        xgb_bayes.maximize(init_points=15, n_iter=2)
    best_params = xgb_bayes.max['params']
    xgb_params = bo_result_to_xgb(best_params)
    return xgb_params


def bo_result_to_xgb(bo_res):
    """将贝叶斯寻参参数转为XGB可接受的参数"""
    xgb_params = bo_res.copy()
    if 'log_gamma' in xgb_params:
        xgb_params['gamma'] = 10 ** xgb_params['log_gamma']
        xgb_params.pop('log_gamma')
    if 'max_depth' in xgb_params:
        xgb_params['max_depth'] = int(np.round(xgb_params['max_depth']))
    if 'max_delta_step' in xgb_params:
        xgb_params['max_delta_step'] = int(np.round(xgb_params['max_delta_step']))
    if 'subsample' in xgb_params:
        xgb_params['subsample'] = max(min(xgb_params['subsample'], 1), 0)
    if 'colsample_bytree' in xgb_params:
        xgb_params['colsample_bytree'] = max(min(xgb_params['colsample_bytree'], 1), 0)
    return xgb_params
