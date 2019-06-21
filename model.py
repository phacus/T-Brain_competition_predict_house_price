import numpy as np
import pandas as pd

train = pd.read_csv('dataset-0510/train.csv')
test = pd.read_csv('dataset-0510/test.csv')

train['parking_price'] = train['parking_price'].fillna(0)

name = test['building_id']

ignored = ['parking_price', 'parking_area', 'txn_floor', 'village_income_median', 'building_id']

train = train.drop(columns=ignored)
test = test.drop(columns=ignored)

feature = ['building_complete_dt', 'land_area', 'building_area', 'town_population',
       'town_population_density', 'IV_MIN', 'VI_MIN', 'VII_1000', 'VII_5000',
       'VII_10000', 'VIII_5000', 'XII_5000', 'XII_10000', 'XIII_5000',
       'XIII_MIN', 'XIV_5000', 'XIV_10000']

train_feature = feature.copy()
train_feature.append('total_price')
train = train[train_feature]

train['total_price'] = np.log1p(train['total_price'])

test = test[feature]

y = train['total_price']
del train['total_price']
X = train.values
y = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# print('Running RandomForestRegressor')
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(n_estimators=200, verbose=2, n_jobs=-1)
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test)*100)

from sklearn.ensemble import GradientBoostingRegressor
print('Running GradientBoostingRegressor')

# n_estimators
n=700

# # self testing ,validating, predicting
# GBR = GradientBoostingRegressor(n_estimators=n, max_depth=4, verbose=2)
# GBR.fit(X_train, y_train)
# print(GBR.score(X_test, y_test)*100)

# predict using test dataset and output submission csv file
GBR = GradientBoostingRegressor(n_estimators=n, max_depth=4, verbose=2)
GBR.fit(X, y)

print('Predicting and outputing the submission file')

Y_pred = GBR.predict(test)
test['total_price'] = np.expm1(Y_pred)
test['building_id'] = name
test[['building_id', 'total_price']].to_csv('output.csv', index=False)