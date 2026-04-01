import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

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

volume_df = pd.read_parquet("data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-04.parquet")
volume_df.index = pd.to_datetime(volume_df.index)

bin_size = 50
window_days = 5
window_hours = window_days * 24

min_price = df_hourly['low'].min()
max_price = df_hourly['high'].max()
bins = np.arange(min_price, max_price + bin_size, bin_size)

times_num = date2num(df_hourly.index)

max_volume = 0
for start in range(0, len(df_hourly), window_hours):
    end = start + window_hours
    df_window = df_hourly.iloc[start:end]
    vol_window = volume_df.loc[df_window.index]
    if df_window.empty or vol_window.empty:
        continue
    vol_bins = np.zeros(len(bins)-1)
    for col in vol_window.columns:
        price_level = float(col)
        idx = np.searchsorted(bins, price_level, side='right') - 1
        if 0 <= idx < len(vol_bins):
            vol_bins[idx] += vol_window[col].sum()
    max_volume = max(max_volume, vol_bins.max())

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(df_hourly.index, df_hourly['close'], color='red', linewidth=1.2, label='Price')

for start in range(0, len(df_hourly), window_hours):
    end = start + window_hours
    df_window = df_hourly.iloc[start:end]
    vol_window = volume_df.loc[df_window.index]
    if df_window.empty or vol_window.empty:
        continue

    vol_bins = np.zeros(len(bins)-1)
    for col in vol_window.columns:
        price_level = float(col)
        idx = np.searchsorted(bins, price_level, side='right') - 1
        if 0 <= idx < len(vol_bins):
            vol_bins[idx] += vol_window[col].sum()

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
ax.set_title("Monthly Price Line with Time-Aligned 5-Day Horizontal Volume Profiles (Scaled by Volume)")
fig.autofmt_xdate()
plt.show()