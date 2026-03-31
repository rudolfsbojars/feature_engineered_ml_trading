import backtrader as bt
import yfinance as yf
from datetime import datetime
from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.vwap import VWAP
from feature_algos.smc import BreakOfStructure, FairValueGap
from feature_algos.volume_profile import VolumeArea, PointOfControl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import matplotlib.dates as mdates

cols = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","trades",
    "taker_base","taker_quote","ignore"
]


class Strategy(bt.Strategy):    
    def __init__(self):
        self.rsi = RSI(self.data, period=14)
        self.structure = AdaptivePIP(self.data)
        self.ema20 = EMA(self.data, period=20)
        self.ema50 = EMA(self.data, period=50)
        self.vwap = VWAP(self.data, period=14)

    def next(self):
        pass

if __name__ == '__main__':
    
    cerebro = bt.Cerebro()

    df = yf.download('EURUSD=X', start='2026-03-1', end='2026-03-30', interval='15m')
    print(df)
    
    df.columns = df.columns.get_level_values(0) 
    df.columns = df.columns.str.lower()
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)


    cerebro.addstrategy(Strategy)
    cerebro.broker.setcash(10000)
    
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    cerebro.run(plot=True)

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    cerebro.plot(style='candlestick')
    
    
    
    
    
"""
    print(pd.read_parquet("data/price_level_volumes/SOLUSDT/vp_SOLUSDT-1h-2024-01.parquet"))

    #plot_volume_profile()

    df = pd.read_csv(
        "data/raw/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2022-02.csv",
        names=cols
    )

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)

    df.set_index("open_time", inplace=True)

    df_1h = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    mpf.plot(
        df_1h,
        type='candle',
        style='charles',
        title='BTCUSDT 1-Hour Candles - Feb 2022',
        ylabel='Price (USD)',
        figsize=(20,10)
    )
    
    
    
def plot_volume_profile():
    df = pd.read_csv(
        "data/raw/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2022-02.csv",
        names=cols
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df.set_index("open_time", inplace=True)

    df_1h = df.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum"
    }).dropna()

    vp = pd.read_parquet("data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-02.parquet")
    vp.index = pd.to_datetime(vp.index)

    hours, price_cols = vp.shape
    all_volume = vp.values 
    price_levels = np.array(vp.columns, dtype=float)

    levels = np.linspace(price_levels.min(), price_levels.max(), 100)
    volume_profile = np.zeros_like(levels)

    for i in range(len(levels)-1):
        mask = (price_levels >= levels[i]) & (price_levels < levels[i+1])
        if mask.any():
            volume_profile[i] = all_volume[:, mask].sum()

    volume_profile_norm = volume_profile / volume_profile.max()

    fig, ax = plt.subplots(figsize=(20,10))

    vp_max_width = 0.3 * len(df_1h) 
    ax.barh(levels, volume_profile_norm*vp_max_width,
            left=-0.5, height=(levels[1]-levels[0]),
            color='lightblue', alpha=0.6, edgecolor=None)

    mpf.plot(df_1h, type='candle', style='charles', ax=ax, volume=False)

    ax.set_xlim(-0.5, len(df_1h)-0.5)
    ax.set_ylabel("Price (USD)")
    ax.set_title("BTCUSDT 1-Hour Candles + Monthly Volume Profile Feb 2022")
    plt.tight_layout()
    plt.show()
    
    
def generate_volume_profile(price_df, volume_df, range):
    pass
    
#takes in a data frame of 1 minute candles and converts them to desired resoltion
def convert_to_time_resoltion(data_frame, time_frame):
    return data_frame.resample(time_frame).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

"""