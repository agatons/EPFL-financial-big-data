# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:38:09 2019

@author: Emil Immonen

Runs the financials investing method. Reads cleaned data files, runs "prepare_data_for_ml.py" 
and "build_ML_model.py" in order to train a gradient boosting model. Additionally, invests
for a given time frame and prints and plots out strategy performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection

import build_MLmodel as mlt
import prepare_data_for_ml as prep
from helper_functions import *

sns.set()

def build_portfolios(i, starting_cap = 100000, yf = True, start = 2010):
    '''
    Parameters
    ----------
    i : Integer
        Random state on how to split train and test data.
    starting_cap : Integer
        How much starting capital you have.
    yf : Use market price from yahoo finance (xxxx-03-01)
    
    Returns
    -------
    portfolio_dic : Dictionary.
        Returns a dictionary of portfolios of stocks for the years 2010-2018.

    '''
    comp_fin, _, y_lr = prep.prepare_data(drop = True, yf = yf)
    
    if yf:
        prices = 'Yf Market price'
    else:
        prices = 'Market price - year end USD'
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(comp_fin, y_lr, test_size = 0.7, random_state=i)
    
    #Row 1249 here, as that is the first row that is years 2013-2010.
    X_train = comp_fin.iloc[1249:]
    y_train = y_lr.iloc[1249:]
    X_test = comp_fin.iloc[:1249]
    y_test = y_lr.iloc[:1249]
    
    model = mlt.train_xgb(X_train, y_train, X_test, y_test, early_stopping = True, plot_train = True)
    
    predictions = model.predict(X_test)
    
    portfolio_dic =  {}
    portfolio_values = []
    portfolio_values.append(starting_cap) 
    
    comp_to_trade = X_test.join(companies[{'Name', 'Ticker'}])
    comp_to_trade = comp_to_trade.join(companies['Year'])
    comp_to_trade['Prediction'] = predictions
    
    #Build a dictionary of portfolios.
    for y in range(start, 2019):
        if y == start:
            comp_to_trade_y = comp_to_trade[comp_to_trade['Year'] == y]
            portfolio_dic[y] = buy_top_x(comp_to_trade_y[{'Name', 'Ticker', 'Prediction', prices}], cash = portfolio_values[y-start])
        else:
            comp_to_trade_y = comp_to_trade[comp_to_trade['Year'] == y]
            portfolio_dic[y] = buy_top_x(comp_to_trade_y[{'Name', 'Ticker', 'Prediction', prices}], cash = portfolio_values[y-start])
    
        portfolio_values.append(calc_end_val(portfolio_dic[y], companies, year = y, yf = yf))
    
    #Plot predictions against true values.
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.hlines(0, min(y_test), max(y_test))
    plt.vlines(0, min(y_test), max(y_test))
    plt.plot(np.arange(-2, max(y_test), 0.1), min(model.evals_result()['validation_1']['rmse'])*np.arange(-2, max(y_test), 0.1), 'r')
    plt.xlabel('True value')
    plt.ylabel('Prediction')
    plt.title('Predictions vs. true returns')
    
    return portfolio_dic, portfolio_values



companies = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')

iters = 1
first_year = 2013

#Build all the portfolios and get the yearly values amount of cash.
portfolios, values = build_portfolios(iters, start = first_year)

#Calculate the return of all the stocks in our datasets equally weighted.
market = [1]
for year in range(first_year, 2019):
    r = companies[companies['Year'] == year]['Yf Return'].mean()
    market.append(market[year-first_year]*(1+r))

#Plot the yearly value of the portfolio compared to the market.
plt.figure()
plt.plot(np.arange(first_year, 2020), np.array(values)/values[0], label = 'Portfolio_'+str(iters))   
plt.plot(np.arange(first_year, 2020), market, label = 'Market')
plt.hlines(market[len(market)-1], xmin = first_year, xmax = 2019, color = 'red')
plt.legend()
plt.title('Cumulative returns compared to market')

#Get monthly closing data for all stocks.
monthly_close = pd.read_pickle('../data/raw/monthly_stock.pkl')['Close']
monthly_close = monthly_close[monthly_close.index.day == 1]

#Calculate and plot the performance of the portfolio.
port_performance(portfolios, first_year, monthly_close)

#%%

'''
Save the yearly portfolios into a csv file
'''

portfolio_table = {}
for year in portfolios.keys():
    portfolio_table[year] = portfolios[year].Name.values

portfolio_table = pd.DataFrame(portfolio_table)
portfolio_table.to_csv('../data/portfolios/portfolio_table.csv')