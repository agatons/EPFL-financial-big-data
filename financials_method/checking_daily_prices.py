# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 22:49:40 2020

@author: Emil
"""

import pandas as pd
import re

data = {}
stocks = pd.read_excel('../data/raw/financials_tickers.xlsx', index_col = 'Unnamed: 0')


for company in stocks['Ticker symbol'].apply(lambda x: re.sub('\..*$', '', x)):
    try:
        data[company] = pd.read_csv('../data/clean/swe_equ/'+company+'.csv')
    except:
        print(company+' missing')
        

start_val = 0
end_val = 0

for company in data.keys():
    if data[company].shape[0] > 0:
        start_val = start_val + data[company]['Close'][0]
        end_val = end_val + data[company]['Close'][data[company].shape[0]-1]
    
end_val/start_val