"""
@author: Emil Immonen

Builds the cleaned dataset into a suitable form for training an ML model
"""

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def format_raw(plot_changes = False):
    '''
    Function reads a raw data file. Renames columns, and reformats the dataframe, so that years are depicted on rows.
    Takes in variable plot_changes, default is False. 
    
    If plot_changes is set to True, plots amount of missing data, in different steps of the cleaning process.

    Returns
    -------
    clean_data : Data Frame
        clean_data has all columns cleaned, but nothing new added. At this point, 
        the raw data file is simply formatted differently.
    '''
    
    data = pd.read_excel('../data/raw/financials_raw_new.xlsx', index_col = 0)
    data.rename(columns = {'Company name Latin alphabet' : 'Name', 'NACE Rev. 2, core code (4 digits)' : 'NACE code'}, inplace = True)
    data.drop(['Consolidation code', 'Country ISO code', 'City'], axis = 1, inplace = True)
    
    #Set figure size and title
    if plot_changes:
        fig, ax = plt.subplots(3)
        fig.set_size_inches(15,10)
        fig.suptitle('Missing data in different files')
    
    #Format and change names for the different financials (the new columns)
    clean_data = data.melt(id_vars = ['Name', 'NACE code', 'Last avail. year'])
    clean_data.sort_values(by = ['Name', 'variable'], inplace = True)
    clean_data['Year'] = clean_data['variable'].apply(lambda x: re.findall('(\d{4})', x)[0])
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\d{4})', '', x))
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\n.*?)(th)', '', x))
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('\n', ' ', x))
    clean_data['variable'] = clean_data['variable'].str.strip()

    #Replace missing values with np.nan
    clean_data.replace('n.a.', np.nan, inplace = True)
    clean_data.replace('n.s.', np.nan, inplace = True)
    
    #Create shape of new dataframe
    null_df = clean_data[clean_data.value.isna()]
    nulls = []
    financials = []
    for financial in clean_data['variable'].unique():
        nulls.append(null_df[null_df['variable'] == financial].shape[0]/
                     clean_data[clean_data['variable'] == financial].shape[0])
        financials.append(financial)
    
    
    #Plot missing values in raw data file
    if plot_changes:
        ax[0].barh(financials, nulls)
        ax[0].set_title('Original')
    
    clean_data.sort_values(by = ['variable', 'Name'], inplace = True)
    tmp = pd.DataFrame()
    
    for cols in clean_data['variable'].unique():
        tmp[cols] = clean_data[clean_data['variable'] == cols].value.values
    
    #Take the rows of shape as this is all companies*years.
    clean_data = clean_data.iloc[0:tmp.shape[0], :]
    clean_data.drop(['variable', 'value'], axis = 1, inplace = True)
    
    #Reshaping data and checking that no extra data went missing
    nulls = []
    clean_data = pd.concat([clean_data.reset_index(), tmp], axis = 1)
    
    for col in financials:
        nulls.append(tmp[tmp[col].isna()].shape[0]/
                     tmp[col].shape[0])
    
    if plot_changes:
        ax[1].barh(financials, nulls)
        ax[1].set_title('Temporary')
    
    nulls = []
    for financial in financials:
        nulls.append(clean_data[clean_data[financial].isna()].shape[0]/
                     clean_data[financial].shape[0])

    if plot_changes:    
        ax[2].barh(financials, nulls)
        ax[2].set_title('New')
    
    clean_data.rename(columns = {'P/L for period [=Net income] USD':'Net income'}, inplace = True)
    
    return clean_data


#Saving data to the cleaned financials file.
def save_data(clean_data):
    clean_data.sort_index(inplace = True)
    clean_data.to_csv('../data/clean/cleaned_financials.csv')

def add_values_to_clean(plot_missing = False):
    '''
    Reads the datafile that was cleaned in format_raw and modifies it by adding additional values.
    Saves the modified data in place of the data file built in clean_raw.
    Adds columns: 'Market price - year(+1) end USD', 'Return', 'Dividend yield' and 'logR'
    
    plot_missing: default false. If set to True, plots % of missing values for different financials.
    
    '''
    years = 10
    data_fin = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
    
    data_fin.drop('Unnamed: 0', axis = 1, inplace = True)

    #Add new shifted column with next years market prices.    
    data_fin['Market price - year(+1) end USD'] = np.nan
    data_fin.sort_values(by = ['Year', 'Name'], inplace = True)
    
    temp = np.array([])
    for year in range(2011, 2020):
        temp = np.append(temp, data_fin[data_fin['Year'] == year]['Market price - year end USD'].values)        
        
    temp = np.append(temp, [np.nan]*int(len(data_fin)/years))
    data_fin['Market price - year(+1) end USD'] = temp
    data_fin.sort_index(inplace = True)
    
    #If there are no dividends change dividends to 0
    data_fin['Dividends per share USD'].fillna(0, inplace = True)
    data_fin['Dividend yield'] = data_fin['Dividends per share USD']/data_fin['Market price - year end USD']
    
    #Add yearly returns
    data_fin['Return'] = (data_fin['Market price - year(+1) end USD']-data_fin['Market price - year end USD'])\
                         /data_fin['Market price - year end USD']
    data_fin['logR'] = np.log(data_fin['Market price - year(+1) end USD']/data_fin['Market price - year end USD'])
    
    
    if plot_missing:
        #Plotting for missing values
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
        
        #Missing values per financial, when all rows with no return are dropped
        dropped = data_fin.dropna(subset = ['Return'])
        ax[2].barh(dropped.iloc[:, 4:dropped.shape[1]].isna().sum().sort_values().index,
                   dropped.iloc[:, 4:dropped.shape[1]].isna().sum().sort_values().values/dropped.shape[0])
        ax[2].set_title('% Missing values per financial, when all nan returns have been dropped.')
    
    data_fin = data_fin[data_fin['Year'] != 2019]
    save_data(data_fin)

def combine_yf_mp():
    data_fin = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
    monthly = pd.read_pickle('../data/raw/monthly_stock.pkl')
    monthly = monthly[monthly['Close']['2CUREX.ST'].index.month == 3]
    monthly = monthly[monthly['Close']['2CUREX.ST'].index.day == 1]
    
    #Read ticker file
    stocks = pd.read_excel('../data/raw/financials_tickers.xlsx', index_col = 'Unnamed: 0')
    stocks['Ticker symbol'] = stocks['Ticker symbol'].apply(lambda x: re.sub('\.', '-', x))
    stocks['Ticker symbol'] = stocks['Ticker symbol'].apply(lambda x: re.sub('$', '.ST', x))
    
    #Add a ticker column that is nan
    data_fin['Ticker'] = np.nan
    
    #Add tickers to data_fin
    for company in stocks['Company name Latin alphabet']:
        change = data_fin[data_fin['Name'] == company].index
        data_fin.loc[change, 'Ticker'] = stocks[stocks['Company name Latin alphabet'] == company]['Ticker symbol'].values
    
    #Melt market prices, so that they can be merged with data_fin
    monthly_close = monthly['Close'].reset_index().melt(id_vars = 'Date', var_name = 'Ticker', value_name = 'Yf Market price')
    monthly_close['Date'] = monthly_close['Date'].apply(lambda x: re.sub('-.*$', '', str(x))).astype(int)
    
    #Merge dataframes
    data_fin = data_fin.reset_index().merge(monthly_close, how = 'left', left_on = ['Year', 'Ticker'],
                                            right_on = ['Date', 'Ticker']).set_index('index')
    data_fin.drop('Date', axis = 1, inplace = True)
    
    #Add new shifted column with next years market prices.
    data_fin.sort_values(by = ['Year', 'Name'], inplace = True)

    temp = np.array([])
    for year in range(2011, 2019):
        temp = np.append(temp, data_fin[data_fin['Year'] == year]['Yf Market price'].values)        
    
    #get values for 2019
    for ticker in data_fin['Ticker'].unique():
        temp = np.append(temp,
                         monthly_close[(monthly_close['Date'] == 2019)&(monthly_close['Ticker'] == ticker)]['Yf Market price'].values[0])
    
    data_fin['Yf Market price - year(+1)'] = temp
    data_fin.sort_index(inplace = True)
    
    #Add yearly returns
    data_fin['Yf Return'] = (data_fin['Yf Market price - year(+1)']-data_fin['Yf Market price'])\
                         /data_fin['Yf Market price']
    data_fin['Yf logR'] = np.log(data_fin['Yf Market price - year(+1)']/data_fin['Yf Market price'])
    
    save_data(data_fin)

def drop_outliers():
    '''
    Function drops abnormally high returns, as it is unclear what is the reason for the returns. 
    In one instance the returns were a result of a stock split.
    '''
    data_fin = pd.read_csv('../data/clean/cleaned_financials.csv', index_col = 'index')
    
    plt.figure()
    plt.scatter(data_fin.index, data_fin['Yf Return'])
    data_fin = data_fin[data_fin['Yf Return'] < 5]
    
    plt.figure()
    plt.scatter(data_fin.index, data_fin['Yf Return'])    
    
    save_data(data_fin)
    
cleaned = format_raw()
save_data(cleaned)
add_values_to_clean()
combine_yf_mp()
drop_outliers()