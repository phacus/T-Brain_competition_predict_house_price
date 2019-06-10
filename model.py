import numpy as np
import pandas as pd

train = pd.read_csv('dataset-0510/train.csv')
test = pd.read_csv('dataset-0510/test.csv')

train_specific_col = train
test_specific = test

name = test['building_id']

del train_specific_col['parking_price']
del train_specific_col['parking_area']
del train_specific_col['txn_floor']
del train_specific_col['village_income_median']
del train_specific_col['building_id']

del test_specific['parking_price']
del test_specific['parking_area']
del test_specific['txn_floor']
del test_specific['village_income_median']
del test_specific['building_id']


y = train['total_price']
del train_specific_col['total_price']
X = train_specific_col.values
y = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# print('Running LinearRegression')
# from sklearn import linear_model
# model = linear_model.LinearRegression()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test)*100)

print('Running RandomForestRegressor')
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, verbose=2, n_jobs=-1)
model.fit(X_train, y_train)
print(model.score(X_test, y_test)*100)

# print('Running GradientBoostingRegressor')
# from sklearn.ensemble import GradientBoostingRegressor
# GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4, verbose=2)
# GBR.fit(X_train, y_train)
# print(GBR.score(X_test, y_test)*100)

Y_pred = model.predict(test_specific)
test['total_price'] = Y_pred
test['building_id'] = name
test[['building_id', 'total_price']].to_csv('first_submission.csv', index=False)