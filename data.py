from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import glob
import os

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

        
        

if __name__ == '__main__':
    process_price_level_volume()