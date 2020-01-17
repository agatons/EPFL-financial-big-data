import pandas as pd
import numpy as np
from src.helpers import *
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import linregress
import trendln
import talib

import src.helpers as helpers


ACTIVE_POSITION = False
LAST_TRADE = 0
STOP_LOSS_PRICE = 0

def data_momentum(df):
    """
    Get the signal data of a stock's dataframe
    
    Input: df -> a stock's dataframe
    
    Returns: the signal data
    """
    
    window_size = 20
    data = df.copy()
    data['ma20'] = talib.SMA(data.Close, timeperiod=20)
    data['ma50'] = talib.SMA(data.Close, timeperiod=50)
    data['ma200'] = talib.SMA(data.Close, timeperiod=200)
    
    data['atr'] = talib.ATR(data.High, data.Low, data.Close)
    
    data['rsi'] = talib.RSI(data.Close, timeperiod=14)
    data.dropna(inplace=True)
    # If data is too sparse
    if data.shape[0] < window_size:
        df['signal'] = 0
        return df
    roll_data = data.iloc[::-1]
    data['signal'] = helpers.roll(roll_data, window_size).apply(momentum_strat)
    # No trading in the "future"
    data['signal'] = data['signal'].shift()
    data.dropna(inplace=True)
    
    return data
    
    
def momentum_strat(df):
    """
    req: ma50 ma200 rsi14
         rolling: 20
    """
    global ACTIVE_POSITION
    global STOP_LOSS_PRICE
        
    # Adjustables
    MA_DIFF = df['ma200'][0] * 0.06
    
    MA20_STRENGTH = 0.3
    MA50_STRENGTH = 0.25
    MA200_STRENGTH = 0
    ATR_STRENGTH = 0.3

    STOP_LOSS_LEVEL = .925
    # Trailing stop loss
    if STOP_LOSS_PRICE < .9 * df.Close[0]:
        STOP_LOSS_PRICE = .9 * df.Close[0]
    
    ma_above = df['ma50'][0] - df['ma200'][0] > MA_DIFF # ma50 above ma200?
    ma_below = df['ma50'][0] - df['ma200'][0] < -MA_DIFF
   
    ma20_slope = helpers.calc_slope(df['ma20'].iloc[::-1])
    ma50_slope = helpers.calc_slope(df['ma50'].iloc[::-1])
    ma200_slope = helpers.calc_slope(df['ma200'].iloc[::-1])
    atr_slope = helpers.calc_slope(df['atr'].iloc[::-1])
    
    # Rsi trend calculations
    rsi_uptrend, rsi_downtrend = helpers.calc_rsi_trend(df.rsi.iloc[::-1])
    price_uptrend, price_downtrend  = helpers.calc_rsi_trend(df.Close.iloc[::-1])
    
    buy_signal = 0
    sell_signal = 0

    if (ma_above and rsi_uptrend and price_uptrend and ma50_slope > MA50_STRENGTH and 
        ma20_slope > MA20_STRENGTH and ma200_slope > MA200_STRENGTH and atr_slope < ATR_STRENGTH):
        buy_signal = 1
    
    # losing momentum, sell
    if (ma_below and ma50_slope < -MA50_STRENGTH and
        rsi_downtrend and price_downtrend) or df.Close[0] < STOP_LOSS_PRICE:
        sell_signal = 1
        
    
    if (sell_signal and ACTIVE_POSITION):
        ACTIVE_POSITION = False
        return -1
    elif (buy_signal and not ACTIVE_POSITION):
        ACTIVE_POSITION = True
        STOP_LOSS_PRICE = df.Close[0] * STOP_LOSS_LEVEL
        return 1
    else:
        return 0



def data_mean_revert(df):
    """
    Get the signal data of a stock's dataframe
    
    Input: df -> a stock's dataframe
    
    Returns: the signal data
    """
    window_size = 5
    data = df.copy()
    data['ma200'] = talib.SMA(data.Close, timeperiod=200)
    
    _, _, data['bol_lower'] = talib.BBANDS(data.Close)
    
    data['rsi3'] = talib.RSI(data.Close, timeperiod=3)
    
    data.dropna(inplace=True)
    # If data is too sparse
    if data.shape[0] < window_size:
        df['signal'] = 0
        return df
    
    roll_data = data.iloc[::-1]
    data['signal'] = helpers.roll(roll_data, window_size).apply(mean_revert_strat)
    # No trading in the "future"
    data['signal'] = data['signal'].shift()
    data.dropna(inplace=True)
    
    return data

def mean_revert_strat(df):
    """
    req: rsi3 bollinger-lower-20 ma200
         rolling: 5
    """
    # time based
    # bollinger bands
    # price > ma200
    # rsi3 < 15
    
    global ACTIVE_POSITION
    global LAST_TRADE
    global STOP_LOSS_PRICE
    
    # adjustables
    RSI_LEVEL = 10
    DAYS_STOP = 12
    BOL_LEVEL = 0.05
    STOP_LOSS_LEVEL = 0.9
    
    buy_signal = df['rsi3'][0] < RSI_LEVEL and df.Close[0] > df['ma200'][0] and \
                 abs(df.Close[0] - df['bol_lower'][0]) < df.Close[0] * BOL_LEVEL
    sell_signal = 0
    
    if ACTIVE_POSITION:
        sell_signal = (pd.to_datetime(df.index[0][0]) - pd.to_datetime(LAST_TRADE)).days >= DAYS_STOP or df.Close[1] < STOP_LOSS_PRICE 
    
    if (buy_signal and not ACTIVE_POSITION):
        LAST_TRADE = pd.to_datetime(df.index[0][0])
        ACTIVE_POSITION = True
        STOP_LOSS_PRICE = df.Close[0] * STOP_LOSS_LEVEL
        return 1
    elif (sell_signal and ACTIVE_POSITION):
        ACTIVE_POSITION = False
        return -1
    else:
        return 0
    
    
def twitter_price_action(df):
    """
    req rolling 9 and ma3 on close
    """
    global ACTIVE_POSITION
    global LAST_TRADE
    c1=df.Close[1] <= df.Open[3]
    c2=df.Low[4] <= df.Close[6]
    c3=df.Open[5] <= df.Open[8]
    c4=df.Close[0] <= df.Open[0]
    
    date = pd.to_datetime(df.index[0][0]).strftime("%A")
    correct_day = 1 if (date == 'Monday' or date == 'Thursday' or date == 'Friday') else 0
    buy_signal = c1 and c2 and c3 and c4 and correct_day
    
    sell_signal1 = df.Close[0] > df.ma3[3]
    sell_signal2 = 0
    """
    if ACTIVE_POSITION:
        sell_signal2 = (pd.to_datetime(df.index[0][0]) - LAST_TRADE).days >= 4"""
    
    if (buy_signal and not ACTIVE_POSITION):
        LAST_TRADE = pd.to_datetime(df.index[0][0])
        ACTIVE_POSITION = True
        return 1
    elif ((sell_signal1 or sell_signal2) and ACTIVE_POSITION):
        ACTIVE_POSITION = False
        return -1
    else:
        return 0
    

def evaluate_strat(df, portfolio_value = 100000):
    """
    Evaluates a strategy.
    
        Input: df -> pandas dataframe with index = [Date, ticker]
            and columns = [Close, signal, nr_of_trades_active]
           
           portfolio_value -> the strating value of the portfolio
           
        Returns: the evaluation dictionary with various evaluation values.
    """
    
    df = df.sort_index()
    liquidity = portfolio_value
    
    num_trades = df.nr_active_trades.max()
    
    trades_result = pd.DataFrame(columns=['portfolio_value','result', 'trade_duration'])
    trades = {}
    dates = df.index.get_level_values(0).unique()
    for date in dates:
        for (_, ticker), (close, signal, _) in df.loc[(date, slice(None)),:].iterrows():
            if signal == 1: # we entered a trade
                num_stocks = ((portfolio_value // num_trades) // close)
                if liquidity > num_stocks * close:
                    liquidity -= num_stocks * close
                else:
                    num_stocks = liquidity // close
                    liquidity -= num_stocks * close
                    
                trades[ticker] = (close, num_stocks)
                
            elif signal == -1:
                num_stocks = trades[ticker][1]
                result = (close - trades[ticker][0]) * num_stocks
                portfolio_value += result
                
                liquidity += num_stocks * close
                trades.pop(ticker)
                trades_result = trades_result.append({'portfolio_value':portfolio_value, 'result':result, \
                                                    'trade_duration':0}, ignore_index=True)
    trades = trades_result
    avg_result = round(trades.result.mean(), 1)
    num_trades = trades.shape[0]
    num_pos_trades = trades[trades.result >= 0].shape[0]
    num_neg_trades = trades[trades.result < 0].shape[0]
    total_gain = round(trades[trades.result >= 0].result.sum(), 1)
    total_loss = round(trades[trades.result < 0].result.sum(), 1)
    gain = round(trades.result.sum(), 1)
    best_trade = round(trades.result.max(), 1)
    worst_trade = round(trades.result.min(), 1)
    time_in_market = trades.trade_duration.sum()
    
    result = {'avg_result':avg_result, 'num_trades':num_trades, 'num_pos_trades':num_pos_trades, 'num_neg_trades':num_neg_trades,
              'total_gain':total_gain, 'total_loss':total_loss, 'gain':gain, 'best_trade':best_trade, 'worst_trade':worst_trade,
              'time_in_market':time_in_market, 'trades':trades, 'buy_and_hold':0, 'result_buy_hold':0}
    
    return result

    
def print_evaluation(dic):
    """
    Prints the evalutation in a nice way to better visualize the performanze of the strategy
    param:
        dic: the evaluation dictionary
    """
    print('Evaluation of strategy')
    print('Gain: {0} kr \nAvg gain: {1} kr/trade'.format(dic['gain'], dic['avg_result']))
    print('    Total gain: {0}'.format(dic['total_gain']))
    print('    Total loss: {0}'.format(dic['total_loss']))
    print('Num trades: {0}'.format(dic['num_trades']))
    print('    Num pos trades: {0}'.format(dic['num_pos_trades']))
    print('    Num neg trades: {0}'.format(dic['num_neg_trades']))
    print('Buy and hold would result in gain: {0}'.format(dic['buy_and_hold']))
    print('    Buy and hold vs gain: {0}'.format(dic['result_buy_hold']))
    if dic['num_trades'] == 0:
        return
    
    plt.figure(figsize=(15,6))
    c = ['#60fc1d', '#f31616']
    plt.subplot(1,4,1)
    plt.pie([dic['num_pos_trades']/dic['num_trades'], dic['num_neg_trades']/dic['num_trades']],
            labels=['Winning', 'Losing'],
            colors=c,
            autopct='%1.1f%%',
            wedgeprops=dict(width=0.5, edgecolor='w'))
    plt.title('%of winning trades')
    
    plt.subplot(1,4,2)
    plt.pie([dic['total_gain']/(abs(dic['total_loss'])+dic['total_gain']),          
             abs(dic['total_loss'])/(abs(dic['total_loss'])+dic['total_gain'])],
             labels=['Gain', 'Loss'],
             colors=c,
             autopct='%1.1f%%',
             wedgeprops=dict(width=0.5, edgecolor='w'))
    plt.title('Gain/Loss ratio')
    
    plt.subplot(1,4,3)
    plt.pie([dic['time_in_market'], 1-dic['time_in_market']],
            labels=['In', 'Out'],
            colors=c,
            autopct='%1.1f%%',
            wedgeprops=dict(width=0.5, edgecolor='w'))
    plt.title('Time in the market')
    
    plt.subplot(1,4,4)
    plt.bar(['gain'], dic['best_trade'], alpha=0.5, color = c[0], edgecolor = 'black')
    plt.bar(['gain'], dic['total_gain']/dic['num_pos_trades'], color = c[0], edgecolor = 'black')
    plt.bar(['loss'], dic['worst_trade'], alpha=0.5, color = c[1], edgecolor = 'black')
    plt.bar(['loss'], dic['total_loss']/dic['num_neg_trades'], color = c[1], edgecolor = 'black')
    plt.title('Average gain/loss and best/worst trade')

    
    plt.show()
           
    
def plot_trades(df, portfolio_value=100000):
    """
    Plots the liquidity and portfolio value per day.
    
    Input: df -> pandas dataframe with index = [Date, ticker]
            and columns = [Close, signal, nr_of_trades_active]
           
           portfolio_value -> the strating value of the portfolio
           
    Returns:
        money_df -> pandas dataframe with date as index and portfolio value and liqudity as columns.
        portfolio_df -> pandas dataframe with dates where signals were generated and which ticker and 
                        close value these signal correspond to
    """
    
    
    min_date = df.index.get_level_values(0).min() - timedelta(days=10)
    max_date = df.index.get_level_values(0).max() + timedelta(days=10)
    index = pd.date_range(min_date, periods=(max_date-min_date).days, freq='D')
    money_df = pd.DataFrame(index=index, columns=['portfolio_value', 'liquidity'])
    money_df['portfolio_value'] = portfolio_value
    money_df['liquidity'] = portfolio_value
    
    portfolio_df = pd.DataFrame(columns=['Date', 'ticker', 'stock_amount', 'position_value']).set_index(['Date', 'ticker'])
    
    num_trades = df.nr_active_trades.max()
    trades = {}
    liquidity = portfolio_value
    dates = df.index.get_level_values(0).unique().sort_values()
    for date in dates:
        for (_, ticker), (close, signal, _) in df.loc[(date, slice(None)),:].sort_values(by='signal', ascending=False).iterrows():
            if signal == 1: # we entered a trade
                num_stocks = ((portfolio_value // num_trades) // close)
                if liquidity <= num_stocks * close:
                    num_stocks = liquidity // close
                liquidity -= num_stocks * close
                money_df.loc[money_df.index >= pd.to_datetime(date), 'liquidity'] = liquidity
                trades[ticker] = (close, num_stocks, date)
                
            elif signal == -1:
                num_stocks = trades[ticker][1]
                result = (close - trades[ticker][0]) * num_stocks
                
                portfolio_value += result
                money_df.loc[money_df.index >= pd.to_datetime(date), 'portfolio_value'] = portfolio_value
                liquidity += num_stocks * close
                money_df.loc[money_df.index > pd.to_datetime(date), 'liquidity'] = liquidity
                
                entry_date = trades[ticker][2]
                investment_dates = pd.date_range(entry_date, periods=(date-entry_date).days+1, freq='D')
                portfolio_df=portfolio_df.append(pd.DataFrame({'Date':investment_dates, 'ticker':ticker,
                                                               'stock_amount':int(num_stocks)}).set_index(['Date', 'ticker']), sort=True)
                
                trades.pop(ticker)
    portfolio_df = portfolio_df.sort_index()
    # Calculate every day portfolio value
    tickers = portfolio_df.index.get_level_values(1).unique()
    for t in tickers:
        t_dates = portfolio_df.loc[(slice(None),t), :].index.get_level_values(0)
        t_df = helpers.get_stock_data(t)
        t_df = t_df[t_df.index.isin(t_dates)]
        t_df = t_df.reindex(t_dates, method='ffill').fillna(0)
        
        assert len(t_df) == len(t_dates)
        portfolio_df.loc[(slice(None),t), 'position_value'] = portfolio_df.loc[(slice(None),t), 'stock_amount'] * t_df.Close.values
        portfolio_df = portfolio_df.sort_index()
        
    money_df['positions_value'] = portfolio_df.groupby('Date').agg({'position_value':'sum'})\
                                    .reindex(money_df.index, fill_value=0).position_value
    money_df['portfolio_value'] = money_df.liquidity + money_df.positions_value
                
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('Portfolio value each date')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    plt.plot(money_df.index, money_df.portfolio_value)
    plt.subplot(1,2,2)
    plt.title('Liquidity each date')
    plt.xlabel('Date')
    plt.ylabel('Liquidity value')
    plt.plot(money_df.index, money_df.liquidity)
    plt.show()
    return money_df, portfolio_df