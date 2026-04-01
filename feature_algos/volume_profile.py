import backtrader as bt
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt

class VolumeProfile(bt.Indicator):
    lines = ('dummy',)
    params = (
        ('bin_size', 50),
        ('window_days', 5),
        ('volume_file', None),
    )

    def __init__(self):
        self.timestamps = []
        self.lows = []
        self.highs = []
        self.closes = []
        self.volumes = []

        self.volume_df = None
        
        if self.p.volume_file:
            self.load_volume_file(self.p.volume_file)

    def next(self):
        ts = self.data.datetime.datetime(0)
        self.timestamps.append(ts)
        self.lows.append(self.data.low[0])
        self.highs.append(self.data.high[0])
        self.closes.append(self.data.close[0])
        self.volumes.append(self.data.volume[0])
        self.lines.dummy[0] = 0 

    def load_volume_file(self, file_path):
        df = pd.read_parquet(file_path)
        self.volume_df = {}
        for ts, row in df.iterrows():
            non_zero = row[row != 0].to_dict()
            if non_zero:
                self.volume_df[ts] = non_zero

    def plot_volume_profile(self):
        if not self.timestamps or not self.volume_df:
            print("No data to plot.")
            return

        df_hourly = pd.DataFrame({
            'low': self.lows,
            'high': self.highs,
            'close': self.closes,
            'volume': self.volumes
        }, index=self.timestamps)

        bin_size = self.p.bin_size
        window_days = self.p.window_days
        window_hours = window_days * 24

        min_price = df_hourly['low'].min()
        max_price = df_hourly['high'].max()
        bins = np.arange(min_price, max_price + bin_size, bin_size)

        # Compute max volume for normalization
        max_volume = 0
        for start in range(0, len(df_hourly), window_hours):
            end = start + window_hours
            df_window = df_hourly.iloc[start:end]
            if df_window.empty:
                continue

            vol_bins = np.zeros(len(bins)-1)
            for ts in df_window.index:
                if ts not in self.volume_df:
                    continue
                for price_str, vol in self.volume_df[ts].items():
                    price_level = float(price_str)
                    idx = np.searchsorted(bins, price_level, side='right') - 1
                    if 0 <= idx < len(vol_bins):
                        vol_bins[idx] += vol
            max_volume = max(max_volume, vol_bins.max())

        fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(df_hourly.index, df_hourly['close'], color='red', label='Price')

        for start in range(0, len(df_hourly), window_hours):
            end = start + window_hours
            df_window = df_hourly.iloc[start:end]
            if df_window.empty:
                continue

            vol_bins = np.zeros(len(bins)-1)
            for ts in df_window.index:
                if ts not in self.volume_df:
                    continue
                for price_str, vol in self.volume_df[ts].items():
                    price_level = float(price_str)
                    idx = np.searchsorted(bins, price_level, side='right') - 1
                    if 0 <= idx < len(vol_bins):
                        vol_bins[idx] += vol

            # Normalize and plot
            vol_ratio = vol_bins / max_volume
            start_time = date2num(df_window.index[0])
            end_time = date2num(df_window.index[-1])
            total_width = end_time - start_time

            for i in range(len(vol_bins)):
                bar_width = total_width * vol_ratio[i]
                ax.barh(
                    y=bins[i],
                    width=bar_width,
                    left=start_time,
                    height=bin_size*0.9,
                    color='lightblue',
                    alpha=0.5,
                    edgecolor='k'
                )

        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Backtrader Price + Horizontal Volume Profile")
        fig.autofmt_xdate()
        plt.show()