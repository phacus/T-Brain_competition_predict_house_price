# coding: utf-8

# import modules
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification

import time

print('Loading data...')

# 讀檔
train = pd.read_csv('dataset-0510/train.csv')

# 設定ignored list(cus training dataset can not be zero or NaN) without Feature Enginnearing
# FE later
ignored = ['parking_price', 'parking_area', 'txn_floor', 'village_income_median', 'building_id']
train = train.drop(columns=ignored)

# 對目標欄位進行處理(cus price range too large)
train['total_price'] = np.log1p(train['total_price'])

# 設定train, test datasets
y = train['total_price']
del train['total_price']
X = train.values
y = y.values
X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2)

# lgb dataset
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# custom metric function
def c_hit(preds, train_data):
    trues = train_data.get_label()
    scores = np.absolute((preds - trues)/trues)
    return 'hit', np.sum(scores), False

print('Start training...')

start_time = time.time()

# Model參數
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': "None",
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'num_threads': -1,
}

# params ={
#         'n_estimators': 10000, 'max_depth' : -1, 'num_leaves' :30,         
#         'objective': 'regression',   'metric': 'rmse',   
#         'learning_rate': 0.01,      'boosting': 'gbdt',     'min_data_in_leaf': 10,
#         'feature_fraction': 0.9,    'bagging_freq':1,       'bagging_fraction': 0.8,     'importance_type': 'gain',
#         'lambda_l1': 0.2,  'subsample': .8,   'colsample_bytree': .9, 'num_threads' : -1
#     }

# CV and training
gbm = lgb.train(
    params, 
    lgb_train, 
    num_boost_round=1000, 
    valid_sets=lgb_eval, 
    early_stopping_rounds=1000,
    # fobj=loglikelihood,
    feval=lambda preds, train_data: [c_hit(preds, train_data)],
    verbose_eval=500,
)

# print('Saving model...')
# gbm.save_model('model.txt')

end_time = time.time()

print('Start predicting...')

# Prediting
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Validating
# 還原目標欄位
y_pred = np.expm1(y_pred)
y_test = np.expm1(y_test)

# 計算hit rate, MAPE
hit = np.absolute((y_test - y_pred)/y_test)
hit_rate = np.sum(hit < 0.1) / len(hit)
MAPE = np.sum(hit)/len(hit)

print('MAPE: ', MAPE)
print('Hit Rate: ', hit_rate * 100,'%')
print('Score: ', hit_rate*(10**4) + (1 - MAPE))
print('Training Time: ', end_time - start_time, 's')