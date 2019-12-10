#!/usr/bin/env python
# coding: utf-8

#Reformatting and cleaning the raw financials data file

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

def format_raw():
    data = pd.read_excel('data/raw/financials_raw_new.xlsx', index_col = 0)
    
    data.rename(columns = {'Company name Latin alphabet' : 'Name', 'NACE Rev. 2, core code (4 digits)' : 'NACE code'}, inplace = True)
    data.drop(['Consolidation code', 'Country ISO code', 'City'], axis = 1, inplace = True)
    
    data.drop(data[data['Last avail. year']!= 2018].index , inplace = True)
    
    fig, ax = plt.subplots(3)
    fig.set_size_inches(15,10)
    
    #Here we change names for the different financials, plot missing values and reshape data
    clean_data = data.melt(id_vars = ['Name', 'NACE code', 'Last avail. year'])
    clean_data.sort_values(by = ['Name', 'variable'], inplace = True)
    clean_data['Year'] = clean_data['variable'].apply(lambda x: re.findall('(\d{4})', x)[0])
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\d{4})', '', x))
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\n.*?)(th)', '', x))
    clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('\n', ' ', x))
    
    clean_data['variable'] = clean_data['variable'].str.strip()
    clean_data.replace('n.a.', np.nan, inplace = True)
    
    
    null_df = clean_data[clean_data.value.isna()]
    nulls = []
    financials = []
    for financial in clean_data['variable'].unique():
        nulls.append(null_df[null_df['variable'] == financial].shape[0]/
                     clean_data[clean_data['variable'] == financial].shape[0])
        financials.append(financial)
    
    ax[0].barh(financials, nulls)
    ax[0].set_title('Original')
    clean_data.sort_values(by = ['variable', 'Name'], inplace = True)
    tmp = pd.DataFrame()
    
    for cols in clean_data['variable'].unique():
        tmp[cols] = clean_data[clean_data['variable'] == cols].value.values
    
    #Take the rows of shape as this is all companies*years.
    clean_data = clean_data.iloc[0:tmp.shape[0], :]
    clean_data.drop(['variable', 'value'], axis = 1, inplace = True)
    
    #reshaping data and checking that no extra data went missing
    nulls = []
    clean_data = pd.concat([clean_data.reset_index(), tmp], axis = 1)
    
    for col in financials:
        nulls.append(tmp[tmp[col].isna()].shape[0]/
                     tmp[col].shape[0])
    
    ax[1].barh(financials, nulls)
    ax[1].set_title('Temporary')
    
    nulls = []
    for financial in financials:
        nulls.append(clean_data[clean_data[financial].isna()].shape[0]/
                     clean_data[financial].shape[0])
    
    ax[2].barh(financials, nulls)
    ax[2].set_title('New')
    return clean_data


def save_data(clean_data):
    #Saving data
    clean_data.sort_index(inplace = True)
    clean_data.drop(clean_data[clean_data['Year'] == '2019'].index, inplace = True)
    
    clean_data.to_csv('data/clean/cleaned_financials.csv')

cleaned = format_raw()
save_data(cleaned)