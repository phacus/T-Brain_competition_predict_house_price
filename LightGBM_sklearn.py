# coding: utf-8

# import modules
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

print('Loading data...')

# 讀檔
train = pd.read_csv('FE_train.csv')

# 設定ignored list(cus training dataset can not be zero or NaN) without Feature Enginnearing
# FE later
# ignored = ['parking_price', 'parking_area', 'txn_floor', 'village_income_median', 'building_id']
# train = train.drop(columns=ignored)

# 對目標欄位進行處理(cus price range too large)
train['total_price'] = np.log1p(train['total_price'])
del train['building_id']

# 設定train, test datasets
y = train['total_price']
del train['total_price']
X = train.values
y = y.values

X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2)

print('Start training...')

def c_hit(y_true, y_pred):
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)
    scores = np.abs((y_pred - y_true)/y_true) > 0.1
    hit_error = np.sum(scores)/len(y_true)
    return 'hit_error', hit_error, False

# Training
gbm = lgb.LGBMRegressor(
    objective='regression', 
    num_leaves=31, 
    learning_rate=0.1, 
    n_estimators=1000, 
    n_jobs=-1, 
    metric="None"
)
gbm.fit(X_train, y_train, 
    eval_set=[(X_test, y_test)], 
    eval_metric=lambda y_true, y_pred: [c_hit(y_true, y_pred)], 
    early_stopping_rounds=100
)

print('Start predicting...')

# Predicting
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

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