# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:33:27 2019

@author: Emil
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(show_plot = False, year = 0):
    data_fin = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
    data_fin.drop({'Capital expenditure per share USD', 'Nominal value'}, axis = 1, inplace = True)
    
    nona = data_fin.dropna(axis = 0)
    
    if year == 0:
        y_r = nona['Return']
        y_logr = nona['logR']
        x_fin = (nona.iloc[:, 4:].astype(float)).drop({'Return', 'Market price - year(+1) end USD', 'logR'}, axis = 1)
    else:
        y_r = nona[nona['Year'] == year]['Return']
        y_logr = nona[nona['Year'] == year]['logR']
        x_fin = (nona[nona['Year'] == year].iloc[:, 4:].astype(float)).drop({'Return', 'Market price - year(+1) end USD', 'logR'}, axis = 1)



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
