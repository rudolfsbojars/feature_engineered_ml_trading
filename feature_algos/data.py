from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import glob
import os
from binance_historical_data import BinanceDataDumper

def import_binance_data():
    dumper = BinanceDataDumper(
        path_dir_where_to_dump="data",
        asset_class="spot",
        data_type="klines",
        data_frequency="1m",
    )

    dumper.dump_data(
        tickers=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        date_start=None,
        is_to_update_existing=False
    )  
        
def normalize_timestamps(folder):
    def normalize(ts):
        ts = int(ts)
        if ts > 10**13:
            return ts // 1000
        return ts

    for root, _, files in os.walk(folder):
        for fname in files:
            if "2025" not in fname and "2026" not in fname:
                continue

            path = os.path.join(root, fname)
            print(f"Processing: {path}")

            new_lines = []

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")

                    if not parts or not parts[0].isdigit():
                        new_lines.append(line.rstrip())
                        continue

                    parts[0] = str(normalize(parts[0]))

                    new_lines.append(",".join(parts))

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
  
def combine_files(root_folder, output_file, year_start=2017, year_end=2027):
    years = [str(y) for y in range(year_start, year_end)]

    with open(output_file, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(root_folder):
            for fname in sorted(files):
                if not any(y in fname for y in years):
                    continue

                path = os.path.join(root, fname)
                print(f"Adding: {path}")

                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)  
                        

def process_file(file_path, save_folder, bin_height=0.1):

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_base","taker_quote","ignore"
    ]

    def candle_to_bins(row):
        low = row.low
        high = row.high
        vol = row.volume

        bins = np.arange(
            np.floor(low / bin_height) * bin_height,
            np.ceil(high / bin_height) * bin_height + bin_height,
            bin_height
        )

        if len(bins) == 0:
            return []

        vol_per_bin = vol / len(bins)
        return [(b, vol_per_bin) for b in bins]

    df = pd.read_csv(file_path, names=cols, on_bad_lines='skip')

    df = df.sort_values("open_time")
    df["minute"] = (df["open_time"] // 60000) * 60000

    rows = []

    for i, row in enumerate(df.itertuples(index=False)):
        if i % 10000 == 0:
            print(f"Processing bar: {row.minute}")

        rows.extend([
            (row.minute, b, v)
            for b, v in candle_to_bins(row)
        ])

    result = pd.DataFrame(rows, columns=["minute", "bin", "volume"])

    result = result.sort_values(["minute", "bin"])

    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(
        save_folder,
        os.path.basename(file_path).replace(".csv", ".parquet")
    )

    result.to_parquet(save_path, index=False)
    
def convert_timeframe(input_path, output_path, compression=15):
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_base","taker_quote","ignore"
    ]
    
    df = pd.read_csv(input_path, names=cols, on_bad_lines='skip')
    df = df.set_index("open_time")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)

    df_resampled = df.resample(f"{compression}min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    df_resampled.index = df_resampled.index.astype("int64") 
    df_resampled.index.name = "open_time"

    df_resampled.to_csv(output_path)
    print(f"Saved {len(df_resampled)} bars to {output_path}")
    
if __name__ == '__main__':
    
    #import_binance_data()
    
    #normalize_timestamps("data/spot/monthly/klines/BTCUSDT/1m")
    #normalize_timestamps("data/spot/monthly/klines/ETHUSDT/1m")
    #normalize_timestamps("data/spot/monthly/klines/SOLUSDT/1m")
    
    #create folder all
    
    #combine_files("data/spot/monthly/klines/BTCUSDT/1m/", "data/spot/all/BTCUSDT-1m-2017-08-2026-03.csv")
    #combine_files("data/spot/monthly/klines/ETHUSDT/1m/", "data/spot/all/ETHUSDT-1m-2017-08-2026-03.csv")
    #combine_files("data/spot/monthly/klines/SOLUSDT/1m/", "data/spot/all/SOLUSDT-1m-2020-08-2026-03.csv")
    
    #create folder volume_levels

    #process_file("data/spot/all/BTCUSDT-1m-2017-08-2026-03.csv", "data/spot/volume_levels", 1)
    #process_file("data/spot/all/ETHUSDT-1m-2017-08-2026-03.csv", "data/spot/volume_levels", 0.1)
    #process_file("data/spot/all/SOLUSDT-1m-2020-08-2026-03.csv", "data/spot/volume_levels", 0.01)
    
    
    #df_timeframe = pd.read_csv("data/spot/all/SOLUSDT-1m-2020-08-2026-03.csv")
    #print("Price Data SOLANA: \n",df_timeframe.tail(50))
    
    #df_volume = pd.read_parquet("data/spot/volume_levels/SOLUSDT-1m-2020-08-2026-03.parquet")
    #print("Volume Level SOLANA: \n",df_volume.tail(50))
    
    convert_timeframe(
        "data/spot/all/ETHUSDT-1m-2017-08-2026-03.csv",
        "data/spot/all/ETHUSDT-15m-2017-08-2026-03.csv",
        compression=15
    )
    
    convert_timeframe(
        "data/spot/all/ETHUSDT-1m-2017-08-2026-03.csv",
        "data/spot/all/ETHUSDT-1h-2017-08-2026-03.csv",
        compression=60
    )
        
    convert_timeframe(
        "data/spot/all/ETHUSDT-1m-2017-08-2026-03.csv",
        "data/spot/all/ETHUSDT-4h-2017-08-2026-03.csv",
        compression=240
    )
    
    pass