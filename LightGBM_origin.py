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

def lgb_train(random_state):
    print('Loading data...')

    # 讀檔
    # train = pd.read_csv('dataset-0510/train.csv')
    train = pd.read_csv('FE_train.csv')
    del train['building_id']

    # 設定ignored list(cus training dataset can not be zero or NaN) without Feature Enginnearing
    # FE later
    # ignored = ['parking_price', 'parking_area', 'txn_floor', 'village_income_median', 'building_id']
    # train = train.drop(columns=ignored)

    # 對目標欄位進行處理(cus price range too large)
    train['total_price'] = np.log1p(train['total_price'])

    # 設定train, test datasets
    y = train['total_price']
    del train['total_price']
    X = train.values
    y = y.values

    # 分割dataset為test, train
    X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    X_test_1 = X_test[:6000]
    y_test_1 = y_test[:6000]
    X_test_2 = X_test[6000:]
    y_test_2 = y_test[6000:]

    # lgb dataset
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval_1 = lgb.Dataset(X_test_1, y_test_1, reference=lgb_train)
    lgb_eval_2 = lgb.Dataset(X_test_2, y_test_2, reference=lgb_train)

    # custom metric function
    def c_hit(preds, train_data):
        trues = train_data.get_label()
        trues = np.expm1(trues)
        preds = np.expm1(preds)
        scores = np.absolute((preds - trues)/trues) > 0.1
        hit_error = np.sum(scores)/train_data.num_data()
        return 'hit_error', hit_error, False

    print('Start training...')

    # Model參數
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': "None",
        'num_leaves': 31,
        # 'max_bin': 512,
        'learning_rate': 0.05,
        # 'min_data_in_leaf': 100,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 1,
        'num_threads': -1,
    }

    # CV and training

    start_time = time.time()

    gbm = lgb.train(
        params, 
        lgb_train, 
        num_boost_round=10000, 
        valid_sets=[lgb_eval_1, lgb_eval_2], 
        early_stopping_rounds=1000,
        feval=lambda preds, train_data: [c_hit(preds, train_data)],
        verbose_eval=100,
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

    # 參數寫入log檔
    # f = open('model_training_log.txt', 'a')
    # log = '\nRandom state: ' + str(random_state) + ', Scores: ' + str(hit_rate*(10**4) + (1 - MAPE))
    # f.write(log)
    # f.close()

    # Create submission csv file
    test = pd.read_csv('FE_test.csv')
    names = test['building_id']
    del test['building_id']
    del test['total_price']
    y_out = gbm.predict(test, num_iteration=gbm.best_iteration)
    test['total_price'] = np.expm1(y_out)
    test['building_id'] = names
    file_name = 'output_' + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + '.csv'
    test[['building_id', 'total_price']].to_csv(file_name, index=False)

lgb_train(11)