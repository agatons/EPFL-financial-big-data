import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
from mpl_finance import candlestick2_ohlc
import matplotlib.dates as mdates
import math


def draw(df, title, intraday=False):

    pd.plotting.register_matplotlib_converters()

    f1, (ax, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize = (15, 10))
    
    candlestick2_ohlc(ax, opens=df['Open'], 
                        highs=df['High'], 
                        lows=df['Low'], 
                        closes=df['Close'], 
                        width=.4, colorup='g', colordown='r')
   

    
    signals = [1,-1]
    for signal in signals:
        ax.scatter(df.reset_index().index[df['signal'] == signal], 
                   df.loc[df['signal'] == signal, 'Low' if signal == 1 else 'High'].values, 
                   label='skitscat', color='blue' if signal == 1 else 'black', 
                   s=125, marker='^' if signal == 1 else 'v')
    
    


    t = pd.to_datetime(df.index).strftime("%Y-%m").values
    p = [""]*(len(t))
    i = 1
    while i < len(t):
        p.insert(i, t[i])
        i += math.floor(len(df)/30)
    # volume plot
    ax2.bar(df.index, df['Volume'])
    

    # Labels
    ymin = df['Low'].min() - 1 # max of (%, -1)
    ymax = df['High'].max() + 1 # min of (%, +1)

    ax.set_ylim([ymin, ymax])

    f1.subplots_adjust(hspace=0)
    # [print(a.get_xticklabels()) for a in f1.axes[:-1]]
    plt.setp([a.get_xticklabels() for a in f1.axes[:-1]], visible=False)
    ax.set_xticklabels(p, rotation=-45)
    ax2.set_xticklabels(p, rotation=-45)
    
    plt.title(title)
    plt.show()
