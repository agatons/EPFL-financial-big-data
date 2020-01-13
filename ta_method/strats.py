import pandas as pd
import numpy as np
from helpers import roll
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.stats import linregress
import trendln
import talib
from imp import reload
import helpers
reload(helpers)
import helpers

ACTIVE_POSITION = False
LAST_TRADE = 0
STOP_LOSS_PRICE = 0

def data_mean_revert(df):
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
    
def data_momentum(df):
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
    """
    MA_DIFF = df['ma200'][0] * 0.03
    
    MA20_STRENGTH = 0.23
    MA50_STRENGTH = 0.15
    MA200_STRENGTH = -0.1
    ATR_STRENGTH = 0.5

    STOP_LOSS_LEVEL = .925
    """
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
    
    # divergence
    #if (rsi_uptrend and price_downtrend and ma20_slope < -MA20_STRENGTH):#price_slope < -PRICE_STRENGTH):
    #    buy_signal = 1
    
    # momentum
    if (ma_above and rsi_uptrend and price_uptrend and ma50_slope > MA50_STRENGTH and ma20_slope > MA20_STRENGTH and ma200_slope > MA200_STRENGTH and atr_slope < ATR_STRENGTH):
        buy_signal = 1
    
    # losing momentum, sell
    if (ma_below and ma50_slope < -MA50_STRENGTH and rsi_downtrend and price_downtrend) or df.Close[0] < STOP_LOSS_PRICE: # A bit wrong with the stop loss price
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

    
def pattern_strat(max_min):
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
    

def evaluate_strat_multiple(df):
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
    if len(df.index) < 5:
        return
    # start value
    portfolio_value = 100000
    liquidity = portfolio_value
    
    num_trades = df.nr_active_trades.max()
    
    # Only keep signal rows and relevant cols    
    df = df[['Close', 'signal', 'tickers']]
    #.drop(df[(df['signal'] == [])].index, axis = 0)
    trades_result = pd.DataFrame(columns=['portfolio_value','result', 'trade_duration'])
    trades = {}
    entered_close = 0
    entered_date = 0
    for index, row in df.iterrows():
        for trade_i in range(len(row['signal'])):
            signal = row['signal'][trade_i]
            ticker = row['tickers'][trade_i]
            close = row['Close'][trade_i]
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
                result = (row['Close'][trade_i] - trades[ticker][0]) * num_stocks
                portfolio_value += result
                
                liquidity += num_stocks * row['Close'][trade_i]
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
    if len(df.index) < 5:
        return
    portfolio_value = 100000
    # Used for calculating time in the market
    time_window = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days
    buy_and_hold = (portfolio_value//df.Close[0])*df.Close[-1]+(portfolio_value%df.Close[0]) - portfolio_value
    # Only keep signal rows and relevant cols    
    df = df.drop(df[(df['signal'] != 1) & (df['signal'] != -1) & (df.index != df.index[-1])].index, axis = 0)[['Close', 'signal']]
    
    trades = pd.DataFrame(columns=['portfolio_value','result', 'trade_duration'])
    active_trade = False
    entered_close = 0
    entered_date = 0
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
            
    if active_trade: # position was still active
        row = df.iloc[-1,:]
        index = df.index[-1]
        
        result = (row['Close'] - entered_close)*(portfolio_value//entered_close)
        portfolio_value += result
        trade_duration = pd.to_datetime(index) - entered_date
        trades = trades.append({'portfolio_value':portfolio_value, 'result':result, \
                                'trade_duration':trade_duration.days}, ignore_index=True)
    
    avg_result = round(trades.result.mean(), 1)
    num_trades = trades.shape[0]
    num_pos_trades = trades[trades.result >= 0].shape[0]
    num_neg_trades = trades[trades.result < 0].shape[0]
    total_gain = round(trades[trades.result >= 0].result.sum(), 1)
    total_loss = round(trades[trades.result < 0].result.sum(), 1)
    gain = round(trades.result.sum(), 1)
    best_trade = round(trades.result.max(), 1)
    worst_trade = round(trades.result.min(), 1)
    time_in_market = trades.trade_duration.sum()/time_window
    
    result = {'avg_result':avg_result, 'num_trades':num_trades, 'num_pos_trades':num_pos_trades, 'num_neg_trades':num_neg_trades,
              'total_gain':total_gain, 'total_loss':total_loss, 'gain':gain, 'best_trade':best_trade, 'worst_trade':worst_trade,
              'time_in_market':time_in_market, 'trades':trades, 'buy_and_hold':buy_and_hold, 'result_buy_hold':gain/buy_and_hold}
    
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
    df['portfolio_value'] = portfolio_value
    active_trade = False
    entered_close = 0
    entered_date = 0
    for index, row in df.iterrows():
        if (not active_trade and row['signal'] == 1): # Entered market
            active_trade = True 
            entered_close = row['Close']
            entered_date = pd.to_datetime(index)

        elif (active_trade and row['signal'] == -1): # Left market
            active_trade = False 
            result = (row['Close'] - entered_close)*(portfolio_value//entered_close)
            portfolio_value += result
            df[df.index >= pd.to_datetime(index)] = portfolio_value
    plt.title('Portfolio value after trades')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    plt.plot(df.index, df.portfolio_value)
    plt.show()
            
    
def plot_trades_multiple(df, portfolio_value=100000):
    num_trades = df.nr_active_trades.max()
    trades = {}
    df['portfolio_value'] = portfolio_value
    liquidity = portfolio_value
    for index, row in df.iterrows():
        for trade_i in range(len(row['signal'])):
            signal = row['signal'][trade_i]
            ticker = row['tickers'][trade_i]
            close = row['Close'][trade_i]
            
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
                result = (row['Close'][trade_i] - trades[ticker][0]) * num_stocks
                portfolio_value += result
                df[df.index >= pd.to_datetime(index)] = portfolio_value
                
                liquidity += num_stocks * row['Close'][trade_i]
                trades.pop(ticker)
    plt.title('Portfolio value after trades')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value')
    plt.plot(df.index, df.portfolio_value)
    plt.show()