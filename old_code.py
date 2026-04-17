
"""

cols = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","trades",
    "taker_base","taker_quote","ignore"
]

def candle_to_bins(row, bin_size):
    low = row.low
    high = row.high
    vol = row.volume
    

    bins = np.arange(
        np.floor(low/bin_size)*bin_size,
        np.ceil(high/bin_size)*bin_size + bin_size,
        bin_size
    )

    if len(bins) == 0:
        return []

    vol_per_bin = vol / len(bins)

    return [(b, vol_per_bin) for b in bins]

    
def process_price_level_volume():
    
    files_og = sorted(glob.glob("data/raw/spot/monthly/klines/SOLUSDT/1m/*.csv"))
    files = [f for f in files_og if ("2025" in f) or ("2026" in f)]

    for file in files:
        df = pd.read_csv(file, names=cols)
        print("Processing:", file)
        
        #convert to types
        df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="us")
        df[["open","high","low","close","volume"]] = df[
            ["open","high","low","close","volume"]
        ].astype(float)
        
        df = df.sort_values("open_time")
        
        df['hour'] = df['open_time'].dt.floor('h')

        rows = []
        for row in df.itertuples():
            rows.extend([(row.hour, b, v) for b, v in candle_to_bins(row, 0.1)])

        temp_df = pd.DataFrame(rows, columns=['hour', 'bin', 'volume'])
        result = temp_df.groupby(['hour', 'bin'])['volume'].sum().unstack(fill_value=0)
        result.index.name = None
        result.columns.name = None

        #saves
        file_name = os.path.basename(file).replace(".csv", "").replace("-1m", "-1h")
        result.to_parquet(f"data/price_level_volumes/SOLUSDT/vp_{file_name}.parquet")

        print(f"Saved {file_name}")
        
        
    
        
        
        
        import backtrader as bt
import pandas as pd
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from datetime import timedelta

# Create teh volume agreagator based on range given
# Range extractor based on type
# Poc Finder
# Volume Area Finder
# Plotter
# Data


class VolumeProfile(bt.Indicator):
    lines = ('dummy',)
    params = (
        ('bin_size', 50),
        ('window_days', 5),
        ('volume_file', None),
    )
    
    # rolling data
    # current range calc
    # current range cycle restart
    # range agregator of data
    # keep track of already computed profile
    # only recompute if min and max is streching so new bin highs and lows

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
                
    def add_volume(self, range_from, range_to):
        return "histogram"

    def plot_volume_profile(self):
        if not self.volume_df:
            print("No volume data to plot.")
            return

        all_timestamps = sorted(self.volume_df.keys())
        lows, highs, closes, volumes = [], [], [], []

        for ts in all_timestamps:
            price_levels = [float(p) for p in self.volume_df[ts].keys()]
            if not price_levels:
                lows.append(0)
                highs.append(0)
                closes.append(0)
                volumes.append(0)
                continue

            low = min(price_levels)
            high = max(price_levels)
            close = price_levels[-1]
            vol = sum(self.volume_df[ts].values())

            lows.append(low)
            highs.append(high)
            closes.append(close)
            volumes.append(vol)

        df_hourly = pd.DataFrame({
            'low': lows,
            'high': highs,
            'close': closes,
            'volume': volumes
        }, index=all_timestamps)

        bin_size = self.p.bin_size
        window_days = self.p.window_days
        window_hours = window_days * 24

        min_price = df_hourly['low'].min()
        max_price = df_hourly['high'].max()
        bins = np.arange(min_price, max_price + bin_size, bin_size)

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
        
        
        
        
        
        """