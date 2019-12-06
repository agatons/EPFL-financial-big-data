#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

data = pd.read_excel('data/raw/financials_raw.xlsx', index_col = 0)

data.rename(columns = {'Company name Latin alphabet' : 'Name', 'NACE Rev. 2, core code (4 digits)' : 'NACE code'}, inplace = True)
data.drop(['Consolidation code', 'Country ISO code', 'City'], axis = 1, inplace = True)

data.drop(data[data['Last avail. year']!= 2018].index , inplace = True)

# In[235]:

#Changing data shape and improving variable names
clean_data = data.melt(id_vars = ['Name', 'NACE code', 'Last avail. year'])
clean_data.sort_values(by = ['Name', 'variable'], inplace = True)
clean_data['Year'] = clean_data['variable'].apply(lambda x: re.findall('(\d{4})', x)[0])
clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\d{4})', '', x))
clean_data['variable'] = clean_data['variable'].apply(lambda x: re.sub('(\n.*?)', '', x))
clean_data['variable'] = clean_data['variable'].str.strip()
clean_data.replace('n.a.', np.nan, inplace = True)


null_df = clean_data[clean_data.value.isna()]
nulls = []
financials = []
for financial in clean_data['variable'].unique():
    nulls.append(null_df[null_df['variable'] == financial].shape[0]/
                 clean_data[clean_data['variable'] == financial].shape[0])
    financials.append(financial)

plt.subplot(3,1,1)
plt.bar(financials, nulls)
plt.xticks(rotation = '45')
plt.title('Original')
clean_data.sort_values(by = ['variable', 'Name'], inplace = True)
tmp = pd.DataFrame()

for cols in clean_data['variable'].unique():
    tmp[cols] = clean_data[clean_data['variable'] == cols].value.values

clean_data = clean_data.iloc[0:5900, :]
clean_data.drop(['variable', 'value'], axis = 1, inplace = True)

# In[236]:
nulls = []

clean_data = pd.concat([clean_data.reset_index(), tmp], axis = 1)

for col in financials:
    nulls.append(tmp[tmp[col].isna()].shape[0]/
                 tmp[col].shape[0])

plt.subplot(3,1,2)
plt.bar(financials, nulls)
plt.xticks(rotation = '45')
plt.title('Temporary')

nulls = []
for financial in financials:
    nulls.append(clean_data[clean_data[financial].isna()].shape[0]/
                 clean_data[financial].shape[0])

plt.subplot(3,1,3)
plt.bar(financials, nulls)
plt.xticks(rotation = '45')
plt.title('New')

# In[195]:
clean_data.sort_index(inplace = True)
clean_data.drop(clean_data[clean_data['Year'] == '2019'].index, inplace = True)
clean_data.to_csv('data/clean/cleaned_financials.csv')

# In[243]:
