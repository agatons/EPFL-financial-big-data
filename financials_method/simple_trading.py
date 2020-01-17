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
import build_MLmodel as mlt
import prepare_data_for_ml as prep
import seaborn as sns

from sklearn import model_selection
from tqdm import tqdm

sns.set()

def buy_random(names_preds, cash, x = 20):
    '''
    Function for testing with random buying. Currently not used. Similar to buy_top_x
    '''
    
    names_preds = names_preds.sample(frac=1)
    buy = pd.DataFrame(names_preds.iloc[:x, :])
    cash_per = cash/x
    
    buy['Shares'] = round(cash_per/(buy['Yf Market price']),3)
    return buy
    
def buy_top_x(names_preds, cash, x = 20):
    '''
    Parameters
    ----------
    names_preds : Dataframe
        Possible stocks to purchase.
    cash : Integer 
        amount of money to spend on stocks 
    x : Integer, optional
        Number of stocks to purchase. The default is 20.
    
    Returns
    -------
    top : DataFrame.
        A DataFrame of the companies to buy and the amount of shares of each.

    '''
    names_preds.sort_values(by = 'Prediction', ascending = False, inplace = True)
    top = pd.DataFrame(names_preds.iloc[:x, :])
    cash_per = cash/x
    
    top['Shares'] = round(cash_per/(top['Yf Market price']),3)
    
    return top
    
def calc_end_val(portf, comp_list, year, yf = True):
    '''
    Parameters
    ----------
    portf : DataFrame
        A portfolio of companies that have been bought.
    comp_list : Dataframe
        All company data.
    year : Integer
        The year when portfolio was bought.

    Returns
    -------
    endval : Integer
        The value of the portfolio at year + 1

    '''
    
    endval = 0
    port_return = []    
    for company in portf['Name']:
        #Market price of company
        if yf:
            mp = comp_list[(comp_list['Year'] == year)&(comp_list['Name'] == company)]\
                       ['Yf Market price - year(+1)'].values     
            mp_before = comp_list[(comp_list['Year'] == year)&(comp_list['Name'] == company)]\
                       ['Yf Market price'].values
            port_return.append(mp/mp_before)
        else:
            mp = comp_list[(comp_list['Year'] == year)&(comp_list['Name'] == company)]\
                       ['Market price - year(+1) end USD'].values            
            #Shares in portoflio
        sc = portf[portf['Name'] == company].Shares.values[0]
        
        if len(mp) < 1:
            endval += 0*sc
        else: endval += mp[0]*sc
            
    return endval

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
    
    
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.hlines(0, min(y_test), max(y_test))
    plt.vlines(0, min(y_test), max(y_test))
    plt.plot(np.arange(-2, max(y_test), 0.1), min(model.evals_result()['validation_1']['rmse'])*np.arange(-2, max(y_test), 0.1), 'r')
    plt.xlabel('True value')
    plt.ylabel('Prediction')
    plt.title('Predictions vs. true returns')
    
    portfolio_dic =  {}
    portfolio_values = []
    portfolio_values.append(starting_cap) 
    
    comp_to_trade = X_test.join(companies[{'Name', 'Ticker'}])
    
    comp_to_trade = comp_to_trade.join(companies['Year'])
    comp_to_trade['Prediction'] = predictions
    

    for y in range(start, 2019):
        if y == start:
            comp_to_trade_y = comp_to_trade[comp_to_trade['Year'] == y]
            portfolio_dic[y] = buy_top_x(comp_to_trade_y[{'Name', 'Ticker', 'Prediction', prices}], cash = portfolio_values[y-start])
        else:
            comp_to_trade_y = comp_to_trade[comp_to_trade['Year'] == y]
            portfolio_dic[y] = buy_top_x(comp_to_trade_y[{'Name', 'Ticker', 'Prediction', prices}], cash = portfolio_values[y-start])
    
        portfolio_values.append(calc_end_val(portfolio_dic[y], companies, year = y, yf = yf))
            

    return portfolio_dic, portfolio_values


def calculate_port_value(year, day = '01', portfolio = 0):
    '''
    Calculates the monthly value of a given portfolio.
    '''
    #DataFrame of the closing prices of each month for given year and portfolio
    temp1 = monthly_close[(monthly_close.index.year == year)&(monthly_close.index.month >= 3)]
    fin_year1 = temp1[temp1.columns & portfolios[year].Ticker]
    temp2 = monthly_close[(monthly_close.index.year == year+1)&(monthly_close.index.month < 3)]
    fin_year2 = temp2[temp2.columns & portfolios[year].Ticker]
    temp = pd.concat([fin_year1, fin_year2])
    
    shares = portfolio.sort_values(by = 'Ticker').Shares
    
    values = np.matmul(temp.values, shares)
    values = dict(zip(temp.index, values))
    return values

def annualized_r(first, last, t = 6):
    '''
    Calculates the annualized return of a portfolio. With default portfolio time = 6.
    '''
    r = ((last/first)+1)**(1/t)
    return r

def sharpe_ratio(rp, sigma, rf = 0):
    ''' 
    Calculate the Sharpe ratio of a portfolio.
    '''
    sharpe = (rp-rf)/sigma
    return sharpe
    
def port_performance():
    '''
    Calculate the performance table of the portfolio with annualized return, volatility and
    Sharpe ratio.
    '''
    #Build a DataFrame where indexes are years and months, and the values are the matching portfolio values.
    monthly_port_value = {}
    for year in range(first_year, 2019):
        monthly_port_value.update(calculate_port_value(year, portfolio = portfolios[year]))
    monthly_port_values = pd.DataFrame(list(monthly_port_value.values()), index = monthly_port_value.keys())
    
    #Monthly logarithmic returns of portfolio
    monthly_logr = np.log(monthly_port_values).diff()
    monthly_r = monthly_port_values.pct_change()
    
    #Annualized return of portfolio
    portfolio_ar = annualized_r(monthly_port_values.iloc[0][0], monthly_port_values.iloc[-1:][0][0])
    
    #Annualized portfolio standard deviation.
    portfolio_astd = np.sqrt(12)*np.std(monthly_r)[0]
    
    #Sharpe ratio of portfolio
    portfolio_sharpe = sharpe_ratio(portfolio_ar, portfolio_astd)
    
    strategy_stats = pd.DataFrame(np.array([[portfolio_ar, portfolio_astd, portfolio_sharpe]]),\
                                  columns = ['Annualized return', 'Std', 'Sharpe ratio'])
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(monthly_port_values)
    plt.title('Monthly value of portfolio')

    
    plt.subplot(2, 1, 2)
    plt.plot(monthly_logr)
    plt.title('Monthly logarithmic returns of portfolio')
    
    
    print(strategy_stats)
    


companies = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
#omxs = yf.Ticker('^OMX')
#omx_quart = omxs.history(period = '10y', interval = '3mo')

#Number of iterations with different training and testing data sets.
iters = 1
first_year = 2013

portfolios, values = build_portfolios(iters, start = first_year)

market = [1]
for year in range(first_year, 2019):
    r = companies[companies['Year'] == year]['Yf Return'].mean()
    market.append(market[year-first_year]*(1+r))

plt.figure()
plt.plot(np.arange(first_year, 2020), np.array(values)/values[0], label = 'Portfolio_'+str(iters))   
plt.plot(np.arange(first_year, 2020), market, label = 'Market')
plt.hlines(market[len(market)-1], xmin = first_year, xmax = 2019, color = 'red')
plt.legend()
plt.title('Cumulative returns compared to market')

monthly_close = pd.read_pickle('../data/raw/monthly_stock.pkl')['Close']
monthly_close = monthly_close[monthly_close.index.day == 1]
port_performance()

#%%

'''
Save the yearly portfolios into a csv file
'''

portfolio_table = {}
for year in portfolios.keys():
    portfolio_table[year] = portfolios[year].Name.values

portfolio_table = pd.DataFrame(portfolio_table)
portfolio_table.to_csv('../data/portfolios/portfolio_table.csv')