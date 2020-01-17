# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:40:41 2020

@author: Emil
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def buy_random(names_preds, cash, x = 20):
    '''
    Function for testing with random buying. Currently not used. Similar to buy_top_x, however buys randomly
    '''
    
    names_preds = names_preds.sample(frac=1)
    buy = pd.DataFrame(names_preds.iloc[:x, :])
    cash_per = cash/x
    
    buy['Shares'] = round(cash_per/(buy['Yf Market price']),3)
    return buy
    
def buy_top_x(names_preds, cash, x = 20):
    '''
    Buy a portfolio of x companies based on the best predictions.

    '''
    names_preds.sort_values(by = 'Prediction', ascending = False, inplace = True)
    top = pd.DataFrame(names_preds.iloc[:x, :])
    cash_per = cash/x
    
    top['Shares'] = round(cash_per/(top['Yf Market price']),3)
    
    return top

def calc_end_val(portf, comp_list, year, yf = True):
    '''
    Calculates the value of the portfolio value in year + 1.
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

def calculate_port_value(year, day = '01', portfolio = 0, monthly_close = 0):
    '''
    Calculates the monthly value of a given portfolio.
    '''
    #DataFrame of the closing prices of each month for given year and portfolio
    temp1 = monthly_close[(monthly_close.index.year == year)&(monthly_close.index.month >= 3)]
    fin_year1 = temp1[temp1.columns & portfolio.Ticker]
    temp2 = monthly_close[(monthly_close.index.year == year+1)&(monthly_close.index.month < 3)]
    fin_year2 = temp2[temp2.columns & portfolio.Ticker]
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

def port_performance(portfolios, first_year, monthly_close):
    '''
    Calculate the performance table of the portfolio with annualized return, volatility and
    Sharpe ratio. Additionally, plots the performance of the strategy.
    '''
    #Build a DataFrame where indexes are years and months, and the values are the matching portfolio values.
    monthly_port_value = {}
    for year in range(first_year, 2019):
        monthly_port_value.update(calculate_port_value(year, portfolio = portfolios[year], monthly_close = monthly_close))
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