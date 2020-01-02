import pandas as pd
import numpy as np
from helpers import roll
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

ACTIVE_POSITION = False
LAST_TRADE = 0

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
    portfolio_value = 10000
    # Used for calculating time in the market
    time_window = (pd.to_datetime(df.index[-1]) - pd.to_datetime(df.index[0])).days
    buy_and_hold = (portfolio_value//df.Close[0])*df.Close[-1]+(portfolio_value%df.Close[0]) - portfolio_value
    # Only keep signal rows and relevant cols
    df = df.drop(df[(df['signal'] != 1) & (df['signal'] != -1)].index, axis = 0)[['Close', 'signal']]
    
    trades = pd.DataFrame(columns=['portfolio_value','result', 'trade_duration'])
    active_trade = False
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
              'time_in_market':time_in_market, 'trades':trades, 'buy_and_hold':buy_and_hold}
    
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