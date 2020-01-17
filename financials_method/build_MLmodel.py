# -*- coding: utf-8 -*-
"""
@author: Emil Immonen
"""

import xgboost as xgb
import matplotlib.pyplot as plt

#Train gradient boosting algorithm
def train_xgb(X_train, y_train, X_test, y_test, early_stopping = False, plot_train = False):
    
    xg_reg = xgb.XGBRegressor(max_depth = 1000, learning_rate = 0.1, n_estimators = 100, objective ='reg:squarederror', 
                              colsample_bytree = 0.5, alpha = 1)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    if early_stopping:
        xg_reg.fit(X_train, y_train, eval_metric = ['rmse'], 
                 eval_set = eval_set, verbose = True, early_stopping_rounds = 50)    
    else:    
        xg_reg.fit(X_train, y_train, eval_metric = ['rmse'], 
                 eval_set = eval_set, verbose = True)    
    
    
    if plot_train:
        plot_training(xg_reg)
        
    return xg_reg

def plot_training(model):
    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(0, epochs)
    
    plt.figure()
    plt.plot(x_axis, results['validation_0']['rmse'], label = 'Train')
    plt.plot(x_axis, results['validation_1']['rmse'], label = 'Test')
    plt.legend()
    plt.title('Model training, rmse')
    
    