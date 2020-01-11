# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:53:37 2019

@author: Emil

Testing machine learning on own data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, model_selection
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import prepare_data_for_ml as prep

x, y_r, y_lr = prep.prepare_data()
y_simple = pd.Series(list(y_r))

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y_lr, test_size = 0.3, random_state = 5)

data_dmatrix = xgb.DMatrix(data=x,label=y_r)

#Try multilinear regression
lin_reg = linear_model.LinearRegression().fit(X_train, y_train)
predictions_lin = lin_reg.predict(X_test)

#Testing random forest
rf_reg = RandomForestRegressor(max_depth = 1000, random_state = 0)
rf_reg.fit(X_train.values, y_train.values)
predictions_rf = rf_reg.predict(X_test)

#Testing xgboost
def train_xgb(X_train, y_train):
    xg_reg = xgb.XGBRegressor(max_depth = 1000, learning_rate = 0.1, n_estimators = 100, objective ='reg:squarederror', 
                              colsample_bytree = 0.5, alpha = 1)
    
    xg_reg.fit(X_train, y_train)
        
    
    #Xgboost with cross validation
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 1,
                    'max_depth': 5, 'alpha': 10,  'prediction' : True,}
    
    cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed = 5)
    
    return xg_reg

def plot_crap():
    xgb_model = train_xgb()
    
    predictions_xgb = xgb_model.predict(X_test)

    
    plt.scatter(y_test.values, predictions_lin, label = 'Linear model', color = 'blue')
    plt.scatter(y_test.values, predictions_rf, label = 'Random forest', color = 'red')
    plt.scatter(y_test.values, predictions_xgb, label = 'XGBoost', color = 'green')
    plt.plot(y_test, np.polyfit(np.array(y_test), np.array(predictions_lin), deg=1)[0]*y_test+
             np.polyfit(np.array(y_test), np.array(predictions_lin), deg=1)[1], color = 'blue')
    plt.plot(y_test, np.polyfit(np.array(y_test), np.array(predictions_rf), deg=1)[0]*y_test+
             np.polyfit(np.array(y_test), np.array(predictions_rf), deg=1)[1], color = 'red')
    plt.plot(y_test, np.polyfit(np.array(y_test), np.array(predictions_xgb), deg=1)[0]*y_test+
             np.polyfit(np.array(y_test), np.array(predictions_xgb), deg=1)[1], color = 'green')
    
    
    
    plt.legend()
    plt.ylabel('predictions')
    plt.xlabel('real')
    plt.yticks(np.arange(min(y_test.values), max(y_test.values)))
    plt.xticks(np.arange(min(y_test.values), max(y_test.values)))
    plt.axhline(0, color = 'black')
    plt.axvline(0, color = 'black')
    
    
    rmse_lin = np.sqrt(mean_squared_error(y_test, predictions_lin))
    rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
    print("RMSE lin: %f\nRMSE rf: %f\nRMSE xgb: %f" % (rmse_lin, rmse_rf, rmse_xgb))
    print("Mean of returns: %f" %(y_test.mean()))

    
#plot_crap()