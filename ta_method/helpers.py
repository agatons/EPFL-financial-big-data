from numpy.lib.stride_tricks import as_strided as stride
import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import argrelextrema
from scipy.stats import linregress
import trendln
import datetime

def create_trade_df(d1='2010-01-01', d2='2020-01-01'):
    start_date = pd.to_datetime(d1)
    end_date = pd.to_datetime(d2)
    index = pd.date_range(start_date, periods=(end_date-start_date).days, freq='D')
    
    columns = ['Close','signal']

    return pd.DataFrame(index=index, columns=columns)

def add_trades(trade_df, signal_df):
    """
    trade_df: df with all trades from all sources
    signal_df: local df for one equity and its signals
    """
    
    # Only keep signal rows and relevant cols    
    signal_df = signal_df.drop(signal_df[(signal_df['signal'] != 1) & 
                                         (signal_df['signal'] != -1)].index, axis = 0)[['Close', 'signal']]
    
    # no trades?
    if signal_df.shape[0] < 2:
        return trade_df
    
    # Remove unfinished trades
    if signal_df.loc[signal_df.index[0], 'signal'] == -1:
        signal_df = signal_df.drop(signal_df.index[0], axis=0)
    
    if signal_df.loc[signal_df.index[-1], 'signal'] == 1:
        signal_df = signal_df.drop(signal_df.index[-1], axis=0)
        
    if signal_df.shape[0] < 2:
        return trade_df
    
    # All trades should get closed
    assert signal_df[signal_df['signal'] == 1].shape[0] == signal_df[signal_df['signal'] == -1].shape[0]
    assert signal_df.shape[0] % 2 == 0
    
    
    for trade_i in range(signal_df.shape[0]//2):
        entry_date = pd.to_datetime(signal_df.index.values[2*trade_i])
        #pd.to_datetime(datetime.datetime.fromordinal(int(signal_df.index.values[2*trade_i])))
        exit_date = pd.to_datetime(signal_df.index.values[2*trade_i+1])
        #pd.to_datetime(datetime.datetime.fromordinal(int(signal_df.index.values[2*trade_i+1])))

        if not np.logical_not(trade_df[(trade_df.index >= entry_date) &
                                       (trade_df.index <= exit_date)]\
                                       .isnull()).values.any():
            # fill dates with active position with 0
            trade_df[(trade_df.index >= entry_date+datetime.timedelta(1)) & (trade_df.index <= exit_date - datetime.timedelta(1))] = 0
            # fill entry/exit date with price and its corresponding flag (1:buy, -1:sell)
            trade_df.loc[entry_date] = (signal_df.Close.values[2*trade_i],1)
            trade_df.loc[exit_date] = (signal_df.Close.values[2*trade_i+1],-1)


    return trade_df
    

def roll(df, w, **kwargs):
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)

def find_max_min(prices):
    prices_ = prices.copy()
    prices_.index = np.linspace(1., len(prices_), len(prices_))
    kr = KernelReg([prices_.values], [prices_.index.values], var_type='c', bw=[1.8])
    f = kr.fit([prices_.index.values])
    smooth_prices = pd.Series(data=f[0], index=prices.index)
    
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    
    price_local_max_dt = []
    for i in local_max:
        if (i>1) and (i<len(prices)-1):
            price_local_max_dt.append(prices.iloc[i-2:i+2].argmax())

    price_local_min_dt = []
    for i in local_min:
        if (i>1) and (i<len(prices)-1):
            price_local_min_dt.append(prices.iloc[i-2:i+2].argmin())
        
    prices.name = 'price'
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['index'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num').price
    
    return max_min


def calc_slope(serie):
    serie = serie.values
    slope, _, _, _, _ = linregress(np.arange(len(serie)), serie) # calulate slope 
    slope = slope/serie[0]*100 #normalize
    return slope

def calc_rsi_trend(rsi_serie):
    # Rsi trend calculations
    
    rsi_serie = rsi_serie.values
    minimaIdxs, maximaIdxs = trendln.get_extrema(rsi_serie)
    
    rsi_uptrend = False
    rsi_downtrend = False
    max_slope = 0
    min_slope = 0
    if minimaIdxs:
        min_values = rsi_serie[minimaIdxs]
        min_slope, min_inter, _, _, _ = linregress(minimaIdxs, min_values) # calulate slope 
    if maximaIdxs:
        max_values = rsi_serie[maximaIdxs]
        max_slope, max_inter, _, _, _ = linregress(maximaIdxs, max_values) # calulate slope 
    
    return max_slope > 0 and min_slope > 0, max_slope < 0 and min_slope < 0
    return round(max_slope, 2), round(min_slope, 2)
    return rsi_uptrend, rsi_downtrend