    # -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:33:27 2019

@author: Emil Immonen

Builds the cleaned dataset into a suitable form for training an ML model
"""

import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(show_plot = False, year = 0, drop = True, yf = True):
    '''
    Parameters
    ----------
    show_plot : BOOLEAN, optional
        DESCRIPTION. The default is False.
    year : INTEGER, optional
        Specifies which year data should be used. The default is 0. If 0, use all years.
    drop : BOOLEAN, True by default.
        If true drops all rows with missing values.    
    yf : BOOLEAN. True by default.
        If true, use market prices from Yahoo Finance (xxxx-03-01), 
        else use market prices from Orbis (xxxx-12-31)
        
    Returns
    -------
    x_fin : financials, aka features needed in machine learning
        DESCRIPTION.
    y_r : labels. the one year return for a stock
        DESCRIPTION.
    y_logr : labels. one year logarithmic return for a stock
        DESCRIPTION.

    '''
    
    data_fin = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
    
    #Drop as they both have a ton of missing values.
    data_fin.drop({'Capital expenditure per share USD', 'Nominal value'}, axis = 1, inplace = True)
    
    #Drop all missing rows with missing values
    
    if yf:
        if drop:
            nona = data_fin.drop({'Return', 'Market price - year(+1) end USD', 'logR'}, axis = 1)
            nona = nona.dropna(axis = 0)
        else:
            nona = data_fin.drop({'Return', 'Market price - year(+1) end USD', 'logR'}, axis = 1)

        if year == 0:
            y_r = nona['Yf Return']
            y_logr = nona['Yf logR']
            x_fin = nona.iloc[:, 4:].drop({'Yf Market price - year(+1)', 'Yf Return', 'Yf logR', 
                                           'Ticker'}, axis = 1)
            x_fin = x_fin.astype(float)
        else:
            y_r = nona[nona['Year'] == year]['Yf Return']
            y_logr = nona[nona['Year'] == year]['Yf logR']
            x_fin = nona[nona['Year'] == year].iloc[:, 4:].drop({'Yf Market price - year(+1)', 'Yf Return', 'Yf logR', 
                                                                 'Ticker'}, axis = 1)
            x_fin = x_fin.astype(float)
    
    else:
        if drop:
            nona = data_fin.dropna(axis = 0)
        else:
            nona = data_fin    

        if year == 0:
            y_r = nona['Return']
            y_logr = nona['logR']
            x_fin = (nona.iloc[:, 4:].astype(float)).drop({'Return', 'Market price - year(+1) end USD', 'logR',
                                                           'Yf Market price - year(+1)', 'Yf Return', 'Yf logR',
                                                           'Ticker'}, axis = 1)
        else:
            y_r = nona[nona['Year'] == year]['Return']
            y_logr = nona[nona['Year'] == year]['logR']
            x_fin = (nona[nona['Year'] == year].iloc[:, 4:].astype(float)).drop({'Return', 'Market price - year(+1) end USD',
                                                                                 'logR', 'Yf Market price - year(+1)', 'Yf Return',
                                                                                 'Yf logR', 'Ticker'}, axis = 1)


    if show_plot:  
        fig, axs = plt.subplots(nrows = 10, ncols = 3)
    
        i = 0
        j = 0
        fig.subplots_adjust(wspace = 0.2, hspace = 1)
    
        for financial in x_fin.columns:
            X=x_fin[financial]
            Y=y_r
            
            axs[i, j].scatter(Y, X,  color='black')
            axs[i, j].set_xlabel('Return')
            axs[i, j].set_title(financial)
            axs[i, j].set_xlim(0, max(Y))
            axs[i, j].set_ylim(0, max(X))
                
            j+=1
            if j == 3:
                i += 1
                j = 0
    
        plt.show()
    
    return x_fin, y_r, y_logr
