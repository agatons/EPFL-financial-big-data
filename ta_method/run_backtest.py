import pandas as pd
import numpy as np
import helpers
import strats
import glob

path = r'../data/clean/swe_equ' # use your path
allfiles = glob.glob(path + "/*.csv")

trade_df = helpers.create_trade_df()

for file_name in np.random.choice(allfiles, 10):
    data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
    signal_data = strats.data_momentum(data)
    
    trade_df = helpers.add_trades(trade_df, signal_data)
    signal_data = strats.data_mean_revert(data)
    
    trade_df = helpers.add_trades(trade_df, signal_data)
    
    
    
strats.plot_trades(trade_df.copy())
result = strats.evaluate_strat(trade_df)
strats.print_evaluation(result)
