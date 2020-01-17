# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 22:36:31 2020

@author: Emil Immonen

Import monthly stock data from yahoo finance and save to a .pkl file

"""
import pandas as pd
import yfinance as yf
import re

stocks = pd.read_excel('../data/raw/financials_tickers.xlsx', index_col = 'Unnamed: 0')
stocks['Ticker symbol'] = stocks['Ticker symbol'].apply(lambda x: re.sub('\.', '-', x))
stocks['Ticker symbol'] = stocks['Ticker symbol'].apply(lambda x: re.sub('$', '.ST', x))

data = yf.download(tickers = list(stocks['Ticker symbol']), start = '2010-03-01', end = '2019-03-01', interval = '1mo')
data.to_pickle('../data/raw/monthly_stock.pkl')