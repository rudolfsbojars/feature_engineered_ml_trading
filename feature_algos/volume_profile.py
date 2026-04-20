import backtrader as bt
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from enum import IntEnum
import pyarrow.dataset as ds

class LookbackWindowMode(IntEnum):
    DAY = 1
    WEEK = 2
    MONTH = 3


class VolumeProfile(bt.Indicator):
    lines = ('dummy',)
    plotinfo = dict(plot=False)
    params = (
        ('level_count_per_range', 50),
        ('lookback_window_mode', LookbackWindowMode.WEEK),
        ('volume_level_file_path', None),
        ('volume_area', 0.7),
    )

    def __init__(self):
        if not self.p.volume_level_file_path:
            raise ValueError("volume_level_file_path is required")

        self.dataset = ds.dataset(self.p.volume_level_file_path, format="parquet")
        
        self.prev_start_ts = None
        self.prev_end_ts = None
        
        self.df_volume = pd.DataFrame(columns=["bin", "volume"])
        
        self.poc = None
        self.va_high = None
        self.va_low = None
        self.poc_delta = None
        self.poc_volume = None

    def next(self):
        current_time = self.data.datetime.datetime(0)
        start_ts, end_ts = self.get_range(current_time)
        
        self.update_data(start_ts,end_ts)
        
        self.structure_volume_profile()
        
        self.prev_start_ts = start_ts
        self.prev_end_ts = end_ts
        
    def structure_volume_profile(self):
        if self.df_volume.empty:
            return
        
        price_low = self.df_volume["bin"].min()
        price_high = self.df_volume["bin"].max()
        
        bin_count = self.p.level_count_per_range
        bin_size = (price_high - price_low) / bin_count
        
        if bin_size == 0:
            return
        
        df = self.df_volume.copy()
        df["level"] = ((df["bin"] - price_low) / bin_size).astype(int).clip(0, bin_count - 1)
        
        levels = pd.DataFrame({
            "level": range(bin_count),
            "price": [price_low + (i * bin_size) for i in range(bin_count)]
        })
        
        aggregated = (
            df.groupby("level", as_index=False)["volume"]
            .sum()
        )
        
        profile = levels.merge(aggregated, on="level", how="left").fillna(0)
        
        poc_idx = profile["volume"].idxmax()
        if(self.poc is None):
            self.poc = profile.loc[poc_idx, "price"]
        
        self.poc_delta = self.poc - profile.loc[poc_idx, "price"]
        self.poc = profile.loc[poc_idx, "price"]
        self.poc_volume = profile.loc[poc_idx, "volume"]
        
        total_volume = profile["volume"].sum()
        target = total_volume * self.p.volume_area
        accumulated = profile.loc[poc_idx, "volume"]
        upper = poc_idx
        lower = poc_idx
        
        while accumulated < target:
            up_vol = profile.loc[upper + 1, "volume"] if upper + 1 < len(profile) else 0
            down_vol = profile.loc[lower - 1, "volume"] if lower - 1 >= 0 else 0
            
            if up_vol == 0 and down_vol == 0:
                break
            
            if up_vol >= down_vol:
                upper = min(upper + 1, len(profile) - 1)
                accumulated += up_vol
            else:
                lower = max(lower - 1, 0)
                accumulated += down_vol
        
        self.va_high = profile.loc[upper, "price"]
        self.va_low = profile.loc[lower, "price"]
        
        self.profile = profile
        
    def update_data(self, start_ts, end_ts):
        if self.prev_start_ts is not None and start_ts == self.prev_start_ts:
            df = self.load_range(self.prev_end_ts + 1, end_ts)
        
            if not df.empty:
                self.df_volume = (
                    pd.concat([self.df_volume, df[["bin", "volume"]]], ignore_index=True)
                    .groupby("bin", as_index=False)["volume"]
                    .sum()
                )
        else:
            df = self.load_range(start_ts, end_ts)
            self.df_volume = (
                df[["bin", "volume"]]
                .groupby("bin", as_index=False)["volume"]
                .sum()
            )
            
    
    def get_range(self, dt):
        mode = self.p.lookback_window_mode

        dt = pd.Timestamp(dt).tz_localize("UTC")

        if mode == LookbackWindowMode.DAY:
            start = dt.normalize()

        elif mode == LookbackWindowMode.WEEK:
            start = dt.normalize() - pd.Timedelta(days=dt.weekday())

        elif mode == LookbackWindowMode.MONTH:
            start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            raise NotImplementedError

        start_ts = int(start.timestamp() * 1000)
        end_ts = int(dt.timestamp() * 1000)
        
        return start_ts, end_ts
    
    def load_range(self, start_ts, end_ts):
        table = self.dataset.to_table(
            filter=(
                (ds.field("minute") >= start_ts) &
                (ds.field("minute") <= end_ts)
            ),
            columns=["minute", "bin", "volume"]
        )

        df = table.to_pandas()
        return df    
        
        
"""
log(O^2)

laods whole file and loops trough on every candle or step the enitre array and recomputes profile everyhtime
the file rows are millions of lines and size for the enitre data set of already precomputed volume bins
is about 200MB which becomes unmanagable

This gets optimized by storing already computed profile, having a rolling file buffer of 2*window days
indexing it bty the timestamp, and not searching for every price level

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

        self.volume_df = {}
        self.window_start = None
        self.window_bins = []

        if self.p.volume_file:
            self.load_volume_file(self.p.volume_file)

        self.poc = None
        self.poc_volume = None
        self.va_low = None
        self.va_high = None
        
        self.prev_poc = None
        self.poc_delta = 0 

    def next(self):
        ts = self.data.datetime.datetime(0)

        self.timestamps.append(ts)
        self.lows.append(self.data.low[0])
        self.highs.append(self.data.high[0])
        self.closes.append(self.data.close[0])
        self.volumes.append(self.data.volume[0])
        self.lines.dummy[0] = 0

        if self.window_start is None:
            self.window_start = ts
            self.window_bins = []

        if ts >= self.window_start + timedelta(days=self.p.window_days):
            self.window_start = ts
            self.timestamps = []
            self.lows = []
            self.highs = []
            self.closes = []
            self.volumes = []
            self.poc = None
            self.va_low = None
            self.va_high = None
            self.window_bins = []

        self.window_bins.append((self.data.low[0], self.data.high[0], self.data.close[0], self.data.volume[0]))

        if not self.window_bins or not self.volume_df:
            return

        lows = [low for low, _, _, _ in self.window_bins]
        highs = [high for _, high, _, _ in self.window_bins]
        min_price = min(lows)
        max_price = max(highs)

        if max_price - min_price < 1e-6:
            return

        bins = np.arange(min_price, max_price + self.p.bin_size, self.p.bin_size)
        vol_bins = np.zeros(len(bins)-1)

        for ts_idx in self.timestamps:
            if ts_idx not in self.volume_df:
                continue
            for price_str, vol in self.volume_df[ts_idx].items():
                price_level = float(price_str)
                idx = np.searchsorted(bins, price_level, side='right') - 1
                if 0 <= idx < len(vol_bins):
                    vol_bins[idx] += vol

        if vol_bins.sum() == 0:
            return

        poc_idx = np.argmax(vol_bins)
        self.poc = (bins[poc_idx] + bins[poc_idx+1])/2
        self.poc_volume = vol_bins[poc_idx]
        total_vol = vol_bins.sum()
        target_vol = 0.76 * total_vol
        
        if self.prev_poc is not None and self.poc is not None:
            if self.poc != self.prev_poc:
                self.poc_delta = self.poc - self.prev_poc
            else:
                self.poc_delta = 0
        else:
            self.poc_delta = 0

        self.prev_poc = self.poc

        va_indices = [poc_idx]
        cum_vol = vol_bins[poc_idx]
        left = poc_idx - 1
        right = poc_idx + 1
        while cum_vol < target_vol:
            left_vol = vol_bins[left] if left >= 0 else -1
            right_vol = vol_bins[right] if right < len(vol_bins) else -1

            if left_vol >= right_vol and left_vol > 0:
                va_indices.append(left)
                cum_vol += left_vol
                left -= 1
            elif right_vol > 0:
                va_indices.append(right)
                cum_vol += right_vol
                right += 1
            else:
                break

        self.va_low = bins[min(va_indices)]
        self.va_high = bins[max(va_indices)+1]

    def load_volume_file(self, file_path):
        df = pd.read_parquet(file_path)
        self.volume_df = {}
        for ts, row in df.iterrows():
            non_zero = row[row != 0].to_dict()
            if non_zero:
                self.volume_df[ts] = non_zero
"""