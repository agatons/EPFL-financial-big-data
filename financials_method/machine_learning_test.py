# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:53:37 2019

@author: Emil

Testing machine learning on own data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, model_selection
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

data_fin = pd.read_csv('data/clean/cleaned_financials.csv')
data_fin.drop(['Unnamed: 0', 'Capital expenditure per share USD', 'Nominal value'], axis = 1, inplace = True)
nona = data_fin.dropna(axis = 0)

y = nona['Return']
x = nona.iloc[:, 6:22].astype(float)
'''
fig, axs = plt.subplots(nrows = 10, ncols = 3)
i = 0
j = 0
fig.subplots_adjust(wspace = 0.2, hspace = 1)

for financial in x.columns:
    X=x[financial]
    Y=y
    
    
    axs[i, j].scatter(X, Y,  color='black')
    axs[i, j].set_xlabel(financial)
    axs[i, j].set_ylabel('Return')
    
    j+=1
    if j == 3:
        i += 1
        j = 0

plt.show()
'''

X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.5, random_state = 5)

reg = linear_model.LinearRegression().fit(X_train, y_train)
predictions = reg.predict(X_test)

rf = RandomForestRegressor(max_depth = 1000, random_state = 0)
rf.fit(X_train.values, y_train.values)

predictions = rf.predict(X_test)
plt.scatter(y_test.values, predictions)
plt.ylabel('predictions')
plt.xlabel('real')
plt.yticks(np.arange(0, max(y_test.values)))
plt.xticks(np.arange(0, max(y_test.values)))