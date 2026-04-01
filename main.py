import backtrader as bt
import yfinance as yf
from datetime import datetime
from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.smc import BreakOfStructure, FVG
from feature_algos.volume_profile import VolumeProfile
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
    def __init__(self, volume_file=None):
        self.rsi = RSI(self.data, period=14)
        self.ema20 = EMA(self.data, period=20)
        self.ema50 = EMA(self.data, period=50)
        self.structure = AdaptivePIP(self.data, atr_period=14, atr_multiplier=5)
        self.bos = BreakOfStructure(pip_indicator=self.structure)
        self.fvg = FVG(self.data, buffer=0.001)
        self.vbp = VolumeProfile(
            self.data,
            bin_size=50,
            window_days=5,
            volume_file=volume_file
        )

    def next(self):
        pass
    
if __name__ == '__main__':
    
    volume_df = pd.read_parquet("data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-04.parquet")
    print(volume_df.head())

    df = pd.read_csv("data/raw/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2022-04.csv", header=None)
    df.columns = [
        "open_time","open","high","low","close","volume_btc",
        "close_time","quote_volume_usd","num_trades",
        "taker_buy_vol_btc","taker_buy_vol_usd","ignore"
    ]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume_btc']]
    df.rename(columns={'volume_btc':'volume'}, inplace=True)

    df_hourly = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_hourly.dropna(inplace=True)

    data = bt.feeds.PandasData(dataname=df_hourly)
    
    volume_df = pd.read_parquet("data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-04.parquet")
    volume_levels= {}
    for ts, row in volume_df.iterrows():
        non_zero = row[row != 0].to_dict()
        if non_zero:  # skip timestamps with all zeros
            volume_levels[ts] = non_zero
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(Strategy, volume_file="data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-04.parquet")

    results = cerebro.run()
    
    cerebro.plot()
    
    
    vp = results[0].vbp
    vp.plot_volume_profile()