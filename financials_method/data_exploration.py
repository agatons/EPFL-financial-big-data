# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 13:54:23 2019

@author: Emil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

years = 9
data_fin = pd.read_csv('data/clean/cleaned_financials.csv')
data_fin.drop('Unnamed: 0', inplace = True, axis = 1)
data_fin.drop('index', inplace = True, axis = 1)

#Adding the yearly return as a financial
data_fin['Market price - year(+1) end USD'] = 'NaN'
data_fin.sort_values(by = ['Year', 'Name'], inplace = True)

temp = np.array([])
for year in range(2011, 2019):
    temp = np.append(temp, data_fin[data_fin['Year'] == year]['Market price - year end USD'].values)
temp = np.append(temp, [np.nan]*int(len(data_fin)/9))
data_fin['Market price - year(+1) end USD'] = temp
data_fin.sort_index(inplace = True)

data_fin['Return'] = data_fin['Market price - year(+1) end USD']/data_fin['Market price - year end USD']

#If there are no dividends change dividends to 0
data_fin['Dividends per share USD'].fillna(0, inplace = True)

fig, ax = plt.subplots(3)

#Missing values per financial
fig.set_size_inches(7, 7)
ax[0].barh(data_fin.iloc[:, 4:data_fin.shape[1]].isna().sum().sort_values().index,
           data_fin.iloc[:, 4:data_fin.shape[1]].isna().sum().sort_values().values/data_fin.shape[0])
ax[0].set_title("% missing values per financial measure")

#Missing values per company
missing_per_row = pd.DataFrame(data_fin['Name'])
missing_per_row['NaNs'] = data_fin.iloc[:, 4:data_fin.shape[1]].isna().sum(axis = 1)

ax[1].plot(missing_per_row.iloc[0:int(data_fin.shape[0]/years),:].index,
           missing_per_row.groupby('Name').sum().sort_values(by = 'NaNs', ascending = False)/(years*data_fin.shape[1]-4))
ax[1].set_title("% Missing values per company")

data_fin.dropna(subset = ['Return'], inplace = True)
ax[2].barh(data_fin.iloc[:, 4:data_fin.shape[1]].isna().sum().sort_values().index,
           data_fin.iloc[:, 4:data_fin.shape[1]].isna().sum().sort_values().values/data_fin.shape[0])

data_fin.to_csv('data/clean/cleaned_financials.csv')