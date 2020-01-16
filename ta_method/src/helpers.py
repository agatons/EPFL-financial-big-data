from numpy.lib.stride_tricks import as_strided as stride
import pandas as pd
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import argrelextrema
from scipy.stats import linregress
import trendln
import datetime

import src.strats as strats

def create_trade_df():
    columns = ['Date', 'ticker','Close','signal', 'nr_active_trades']
    df = pd.DataFrame(columns=columns)
    df = df.set_index(['Date', 'ticker']) # Init multi index
    
    return df

def get_trade_df(signal_dfs, max_nr_active_trades = 12, d1='2010-01-01', d2='2020-01-01'):
    """
    trade_df: df with all trades from all sources
    signal_df: local df for one equity and its signals
    """
    trade_df = create_trade_df()
    # Setup active trades series
    start_date = pd.to_datetime(d1)
    end_date = pd.to_datetime(d2)
    index = pd.date_range(start_date, periods=(end_date-start_date).days, freq='D')
    active_trades_ser = pd.Series(index=index, data=0)
    
    # Go through all signal dataframes
    for signal_df, ticker in signal_dfs:
        trade_df = trade_df.sort_index()
        # Only keep signal rows and relevant cols  
        signal_df = signal_df[signal_df.signal != 0][['Close', 'signal']]

        # no trades?
        if signal_df.shape[0] < 2:
            continue

        # Remove unfinished trades
        if signal_df.loc[signal_df.index[0], 'signal'] == -1:
            signal_df = signal_df.drop(signal_df.index[0], axis=0)

        if signal_df.loc[signal_df.index[-1], 'signal'] == 1:
            signal_df = signal_df.drop(signal_df.index[-1], axis=0)

        # no trades?
        if signal_df.shape[0] < 2:
            continue

        # All trades should get closed
        assert signal_df[signal_df['signal'] == 1].shape[0] == signal_df[signal_df['signal'] == -1].shape[0]
        assert signal_df.shape[0] % 2 == 0

        for trade_i in range(signal_df.shape[0]//2):
            entry_date = pd.to_datetime(signal_df.index.values[2*trade_i])
            exit_date = pd.to_datetime(signal_df.index.values[2*trade_i+1])
            active_trades = active_trades_ser[(active_trades_ser.index >= entry_date) & (active_trades_ser.index <= exit_date)].max()
            
            if active_trades < max_nr_active_trades: # theres room for one more trade
                 # Increment active trades
                active_trades_ser[(active_trades_ser.index >= entry_date) & (active_trades_ser.index <= exit_date)] += 1  

                active_trades_entry = active_trades_ser[active_trades_ser.index == entry_date]
                active_trades_exit = active_trades_ser[active_trades_ser.index == exit_date]

                # Add rows
                trade_df.loc[(entry_date, ticker), :] = (signal_df.Close.values[2*trade_i], 1, active_trades_entry)
                trade_df.loc[(exit_date, ticker), :] = (signal_df.Close.values[2*trade_i+1], -1, active_trades_exit)
            
    return trade_df

def get_stock_data(ticker_name, path = r'../data/clean/swe_equ'):
    file_name = path + '/' + ticker_name + '.csv'
    return pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])

def evaluate_saved_trades(filename):
    trade_df = pd.read_csv(filename, parse_dates=['Date']).set_index(['Date','ticker']).sort_index()
    money_df, portfolio_df = strats.plot_trades_multiple(trade_df)
    result = strats.evaluate_strat_multiple(trade_df)
    strats.print_evaluation(result)
    return trade_df, money_df, result

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

    
def get_patterns(max_min):
    # Not like the others
    # https://www.quantopian.com/posts/an-empirical-algorithmic-evaluation-of-technical-analysis
    patterns = {}
    patterns['HS'] = []
    patterns['IHS'] = []
    patterns['BTOP'] = []
    patterns['BBOT'] = []
    patterns['TTOP'] = []
    patterns['TBOT'] = []
    patterns['RTOP'] = []
    patterns['RBOT'] = []
    
    for i in range(5, len(max_min)):
        window = max_min.iloc[i-5:i]

        # pattern must play out in less than 36 days
        if window.index[-1] - window.index[0] > 35:
            continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        rtop_g1 = np.mean([e1,e3,e5])
        rtop_g2 = np.mean([e2,e4])
        # Head and Shoulders
        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
            (abs(e1 - e5) <= 0.03*np.mean([e1,e5])) and \
            (abs(e2 - e4) <= 0.03*np.mean([e1,e5])):
                patterns['HS'].append((window.index[0], window.index[-1]))

        # Inverse Head and Shoulders
        elif (e1 < e2) and (e3 < e1) and (e3 < e5) and \
            (abs(e1 - e5) <= 0.03*np.mean([e1,e5])) and \
            (abs(e2 - e4) <= 0.03*np.mean([e1,e5])):
                patterns['IHS'].append((window.index[0], window.index[-1]))

        # Broadening Top
        elif (e1 > e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['BTOP'].append((window.index[0], window.index[-1]))

        # Broadening Bottom
        elif (e1 < e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['BBOT'].append((window.index[0], window.index[-1]))

        # Triangle Top
        elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['TTOP'].append((window.index[0], window.index[-1]))

        # Triangle Bottom
        elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['TBOT'].append((window.index[0], window.index[-1]))

        # Rectangle Top
        elif (e1 > e2) and (abs(e1-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e3-rtop_g1)/rtop_g1 < 0.0075) and (abs(e5-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e2-rtop_g2)/rtop_g2 < 0.0075) and (abs(e4-rtop_g2)/rtop_g2 < 0.0075) and \
            (min(e1, e3, e5) > max(e2, e4)):

            patterns['RTOP'].append((window.index[0], window.index[-1]))

        # Rectangle Bottom
        elif (e1 < e2) and (abs(e1-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e3-rtop_g1)/rtop_g1 < 0.0075) and (abs(e5-rtop_g1)/rtop_g1 < 0.0075) and \
            (abs(e2-rtop_g2)/rtop_g2 < 0.0075) and (abs(e4-rtop_g2)/rtop_g2 < 0.0075) and \
            (max(e1, e3, e5) > min(e2, e4)):
            patterns['RBOT'].append((window.index[0], window.index[-1]))
            
    return patterns