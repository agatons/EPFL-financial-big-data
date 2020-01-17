import datetime
import pandas as pd
import numpy as np
import src.helpers as helpers
import src.strats as strats
import glob
import dask
import seaborn as sns
dask.config.set(scheduler='processes')

path = r'../data/clean/swe_equ' # use your path
allfiles = glob.glob(path + "/*.csv")

BOOTSTRAP_AMOUNT = 40
BOOTSTRAP_SET_SIZE = 50

def get_signal_data(file_name):
    data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
    return strats.data_momentum(data)

worker_delayed = dask.delayed(get_signal_data)

money_dfs = []

for i in range(BOOTSTRAP_AMOUNT):
    allpromises = [(worker_delayed(fn), fn[len(path)+1:-4]) for fn in np.random.choice(allfiles, BOOTSTRAP_SET_SIZE, replace=False)]

    all_dfs = dask.compute(allpromises)

    trade_df = helpers.get_trade_df(all_dfs[0])
    money_df = strats.get_portfolio_value(trade_df)
    money_dfs.append(money_df)

port_values = [df.portfolio_value for df in money_dfs]
port_values_df = pd.concat(port_values, ignore_index=True, join='outer', axis=1)
port_values_df.index = pd.to_datetime(port_values_df.index)

plt.figure(figsize=(8,5))
ax = sns.distplot(port_values_df.ffill().iloc[-1], norm_hist=True)
plt.axvline(x=port_values_df.iloc[-1].mean(), color='r')
plt.legend(['Average value'])
plt.title('Distribution of final portfolio value', fontsize=14)
plt.xlabel('Portfolio value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

plt.figure(figsize=(8,5))
port_values_df['mean_value'] = port_values_df.mean(axis=1)

for (columnName, columnData) in port_values_df.iteritems():
    plt.plot(columnData, color='b', alpha=0.2)
    
plt.plot(port_values_df.sort_index().mean_value, color='r', label='Average portfolio')
plt.title('Portfolio value over time for each bootstrap', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio value', fontsize=12)
plt.legend()
plt.show()

port_values_df['year'] = port_values_df.index.year
r = np.log(port_values_df.groupby('year').last().mean_value).diff().dropna()

annualized_return = ((port_values_df.mean_value.iloc[-1])/100000) ** (1/9)
std = r.std()
sharpe_ratio = (((port_values_df.mean_value.iloc[-1])/100000) ** (1/9) - 1) / r.std()

print(annualized_return, std, sharpe_ratio)
