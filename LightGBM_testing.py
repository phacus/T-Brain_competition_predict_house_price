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
import gc

def lgb_train(random_state, saving, logging):
    print('Loading data...')

    # 讀檔
    # train = pd.read_csv('dataset-0510/train.csv')
    train = pd.read_csv('FE_train.csv')
    test = pd.read_csv('FE_test.csv')
    del train['building_id']

    # Feature Engineering
    print('Start Feature Engineering...')
    target_df = train.groupby(['city', 'town']).agg({'building_area' : ['mean', 'median'], 'land_area' : ['mean', 'median'], 'total_price' : ['mean', 'median']}).reset_index()
    target_df.columns = [i[0] + '_' + i[1]  if i[1] != '' else i[0] for i in target_df.columns.tolist()]
    target_df['price_land_rate_median'] = np.log1p(target_df['total_price_median']) / target_df['land_area_median']
    target_df['price_building_rate_median'] = np.log1p(target_df['total_price_median']) / target_df['building_area_median']
    target_df['price_land_rate_mean'] = np.log1p(target_df['total_price_mean']) / target_df['land_area_mean']
    target_df['price_building_rate_mean'] = np.log1p(target_df['total_price_mean']) / target_df['building_area_mean']

    combine_cols = ['city', 'town', 'price_land_rate_median', 'price_building_rate_median', 'price_land_rate_mean', 'price_building_rate_mean']
    train = pd.merge(train, target_df[combine_cols], on =['city', 'town'], how='left')
    test = pd.merge(test, target_df[combine_cols], on =['city', 'town'], how='left')

    train.loc[train['building_area'] == 4, 'parking_area'] = train.loc[train['building_area'] == 4, 'building_area'] / train.loc[train['building_area'] == 4, 'total_floor']
    test.loc[train['building_area'] == 4, 'parking_area'] = test.loc[test['building_area'] == 4, 'building_area'] / test.loc[test['building_area'] == 4, 'total_floor']
    drop_cols = [i for i in train.columns if np.sum(train[i]) == 60000 and 'index' in i]

    train.drop(['town'], axis = 1, inplace = True)
    test.drop(['town'], axis = 1, inplace = True)
    train.drop(drop_cols, axis = 1, inplace = True)
    test.drop(drop_cols, axis = 1, inplace = True)
    gc.collect()

    print(train.shape, test.shape)
    del test['total_price']

    # 對目標欄位進行處理(cus price range too large)
    train['total_price'] = np.log1p(train['total_price'])

    # 設定train, test datasets
    y = train['total_price']
    del train['total_price']
    X = train.values
    y = y.values

    # 分割dataset為test, train
    X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    valid_split_size = int(X_test.shape[0]/2)
    
    X_test_1 = X_test[:valid_split_size]
    y_test_1 = y_test[:valid_split_size]
    X_test_2 = X_test[valid_split_size:]
    y_test_2 = y_test[valid_split_size:]

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

    # Model參數
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': "None",
        'num_leaves': 31,
        # 'max_bin': 512,
        'learning_rate': 0.025,
        # 'min_data_in_leaf': 100,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': 1,
        'num_threads': -1,
    }

    categorical_feature = ['building_material', 'building_type', 'building_use',
     'parking_way', 'parking_way']

    feature_name = [i for i in train.columns]

    # CV and training

    # print('Loading previous model...')
    # bst = lgb.Booster(model_file='model.txt')

    print('Start training...')

    start_time = time.time()

    gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=100000, 
        valid_sets=[lgb_eval_1, lgb_eval_2], 
        early_stopping_rounds=10000,
        feval=lambda preds, train_data: [c_hit(preds, train_data)],
        verbose_eval=1000,
        # init_model=bst,
        # feature_name=feature_name,
        # categorical_feature=categorical_feature,
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
    score = hit_rate*(10**4) + (1 - MAPE)

    print('MAPE: ', MAPE)
    print('Hit Rate: ', hit_rate * 100,'%')
    print('Score: ', score)
    print('Training Time: ', end_time - start_time, 's')

    # 參數寫入log檔
    if logging:
        f = open('model_training_log.txt', 'a')
        log = '\nRandom state: ' + str(random_state) + ', Scores: ' + str(score) + ', ' + str(time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()))
        f.write(log)
        f.close()

    # Create submission csv file
    if saving or score > 5750:
        names = test['building_id']
        del test['building_id']
        # del test['total_price']
        y_out = gbm.predict(test, num_iteration=gbm.best_iteration)
        test['total_price'] = np.expm1(y_out)
        test['building_id'] = names
        file_name = 'output_' + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + '.csv'
        directory = 'Submission/'
        test[['building_id', 'total_price']].to_csv(directory + file_name, index=False)

lgb_train(random_state=31, saving=False, logging=False)