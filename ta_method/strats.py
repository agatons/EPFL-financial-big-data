import pandas as pd
import numpy as np
from helpers import roll
from datetime import datetime

ACTIVE_POSITION = False

def evaluate_strat(df):
    # always go all in in positions
    """
    avg gain/loss
    nr of trades
    nr of +/-
    total gain/loss
    gain
    time in market
    best trade
    worst trade
    -max drawdown
    -max runup
    """
    # Only keep signal rows and relevant cols
    time_window = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days
    df = df.drop(df[(df['signal'] != 1) & (df['signal'] != -1)].index, axis = 0)[['Close', 'signal']]
    
    trades = pd.DataFrame(columns=['portfolio_value','result', 'trade_duration'])
    active_trade = False
    portfolio_value = 10000
    for index, row in df.iterrows():
        if (not active_trade and row['signal'] == 1): # Entered market
            active_trade = True 
            entered_close = row['Close']
            entered_date = pd.to_datetime(index)

        elif (active_trade and row['signal'] == -1): # Left market
            active_trade = False 
            result = (row['Close'] - entered_close)*(portfolio_value//entered_close)
            portfolio_value += result
            trade_duration = pd.to_datetime(index) - entered_date
            trades = trades.append({'portfolio_value':portfolio_value, 'result':result, \
                                    'trade_duration':trade_duration.days}, ignore_index=True)
    
    avg_result = trades.result.mean()
    num_trades = trades.shape[0]
    num_pos_trades = trades[trades.result >= 0].shape[0]
    num_neg_trades = trades[trades.result < 0].shape[0]
    total_gain = trades[trades.result >= 0].result.sum()
    total_loss = trades[trades.result < 0].result.sum()
    gain = trades.result.sum()
    best_trade = trades.result.max()
    worst_trade = trades.result.min()
    time_in_market = trades.trade_duration.sum()/time_window
    
    result = {'avg_result':avg_result, 'num_trades':num_trades, 'num_pos_trades':num_pos_trades, 'num_neg_trades':num_neg_trades,
              'total_gain':total_gain, 'total_loss':total_loss, 'gain':gain, 'best_trade':best_trade, 'worst_trade':worst_trade,
              'time_in_market' : time_in_market, 'trades':trades}
    
    return result
 
def ma_strat(df):
    global ACTIVE_POSITION
    close_sum = df.Close.sum()
    ma200_sum = df.ma200.sum()
    if (close_sum>ma200_sum and not ACTIVE_POSITION):
        # Buy
        ACTIVE_POSITION = True
        return 1
    elif (close_sum<ma200_sum and ACTIVE_POSITION):
        # Sell
        ACTIVE_POSITION = False
        return -1
    else:
        return 0