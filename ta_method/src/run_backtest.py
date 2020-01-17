import datetime
import pandas as pd
import numpy as np
import src.helpers as helpers
import src.strats as strats
import glob
import dask
dask.config.set(scheduler='processes')

path = r'../data/clean/swe_equ' # use your path
allfiles = glob.glob(path + "/*.csv")

def get_signal_data(file_name):
    data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
    return strats.data_momentum(data)

worker_delayed = dask.delayed(get_signal_data)

allpromises = [(worker_delayed(fn), fn[len(path)+1:-4]) for fn in allfiles[0:10]]

all_dfs = dask.compute(allpromises)

trade_df = helpers.get_trade_df(all_dfs[0])

strats.plot_trades(trade_df.copy())
result = strats.evaluate_strat(trade_df)
strats.print_evaluation(result)
