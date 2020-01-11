# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:38:09 2019

@author: Emil
"""

import pandas as pd
import yfinance as yf
import machine_learning_test as mlt
import prepare_data_for_ml as prep
import re
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

companies = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
omxs = yf.Ticker('^OMX')
omx_quart = omxs.history(period = '10y', interval = '3mo')

tickers = pd.read_excel('../data/raw/financials_tickers.xlsx', index_col = 'Unnamed: 0')
tickers['Ticker symbol'] = tickers['Ticker symbol'].apply(lambda x: re.sub('\.', '-', x))
tickers['Ticker symbol'] = tickers['Ticker symbol'].apply(lambda x: re.sub('$', '.ST', x))

def invest_portfolio(i):
    
    stocks = 20
    comp_fin, _, y_lr = prep.prepare_data()
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(comp_fin, y_lr, test_size = 0.7, random_state=i)
    
    model = mlt.train_xgb(X_train, y_train)
    predictions = model.predict(X_test)
    
    
    def buy_top_20(names_preds):
        names_preds.sort_values(by = 'Prediction', ascending = False, inplace = True)
        top20 = pd.DataFrame(names_preds.iloc[:stocks, :])
        print(top20)
        return top20
    
    
    def calc_end_val(portf, comp_list, year1):
        endval = 0
        for company in portf['Name']:

            mp = comp_list[(comp_list['Year'] == year1)&(comp_list['Name'] == company)]\
                       ['Market price - year end USD'].values

            
            sc = portf[portf['Name'] == company].Shares.values[0]

            if len(mp) != 1:
                endval += 0*sc
            else: endval += mp[0]*sc
    
        return endval
    
    
    portfolio =  {}
    portfolio_value = []
    comp_fin = X_test.join(companies['Name'])
    comp_fin = comp_fin.join(companies['Year'])
    comp_fin['Prediction'] = predictions
    
    for y in range(2010, 2019):
        if y > 2010:
            value_end = calc_end_val(portfolio[y-1], companies, y)
            
        else:
            value_end = 100000
    
        portfolio_value.append(value_end)
        
        comp_fin_y = comp_fin[comp_fin['Year'] == y]
        comp_fin_y['Shares'] = (value_end/stocks/comp_fin_y['Market price - year end USD']).astype(int)
        
        portfolio.update({y : buy_top_20(comp_fin_y[{'Name', 'Prediction', 'Market price - year end USD', 'Shares'}])})
        
        
    return portfolio_value


# setup toolbar
iters = 1

for i in tqdm(range(iters)):
    portfolio_val= invest_portfolio(i)
    plt.plot(np.arange(2010, 2019), np.array(portfolio_val)/portfolio_val[0])


plt.plot(np.arange(2010, 2019), omx_quart.Close.iloc[1:36:4]/omx_quart.Close.iloc[1], label = 'Index', color = 'black')
plt.hlines(omx_quart.Close.iloc[29]/omx_quart.Close.iloc[1], xmin = 2010, xmax = 2019, color = 'red')
plt.legend()
plt.show()