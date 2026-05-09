import backtrader as bt
from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.smc import BreakOfStructure, FVG
from feature_algos.volume_profile import VolumeProfile, LookbackWindowMode
import pandas as pd
import os
from pathlib import Path
import glob

cols = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","trades",
    "taker_base","taker_quote","ignore"
]

class FeatureExtractedStrategy(bt.Strategy):    
    def __init__(self, volume_level_file_path=None, feature_save_file_path=None):
        self.rsi = RSI(self.data, period=14, upper=70, lower=30)
        self.ema20 = EMA(self.data, period=20)
        self.ema50 = EMA(self.data, period=50)
        self.pip = AdaptivePIP(self.data, atr_period=14, atr_multiplier=5)
        self.bos = BreakOfStructure(self.data, lookback=50, pip_indicator=self.pip)
        
        self.fvg = FVG(self.data)
        self.vbp = VolumeProfile(
            self.data,
            level_count_per_range=50,
            lookback_window_mode=LookbackWindowMode.WEEK,
            volume_level_file_path=volume_level_file_path,
            volume_area=0.7,
        )
        
        self.feature_save_file_path = feature_save_file_path

        self.feature_rows = []
        
                
    def next(self):
        self.save_values()
        pass
    
    def stop(self):
        df = pd.DataFrame(self.feature_rows)
        df.to_csv(self.feature_save_file_path, index=False) 
        print(f"Saved {len(df)} rows to features csv")
    
    def save_values(self):
        timestamp = self.data.datetime.datetime(0)
        print("Processed at: ", timestamp)

        open_ = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        volume = self.data.volume[0]

        rsi = self.rsi.lines.rsi[0]
        ema20 = self.ema20.lines.ema[0]
        ema50 = self.ema50.lines.ema[0]

        bos_bull = self.bos.lines.bos_bull[0] if hasattr(self.bos.lines, 'bos_bull') else float('nan')
        bos_bear = self.bos.lines.bos_bear[0] if hasattr(self.bos.lines, 'bos_bear') else float('nan')

        fvg_up_active = self.fvg.fvg_up_active[0]
        fvg_down_active = self.fvg.fvg_down_active[0]
        fvg_up_price = self.fvg.fvg_up[0]
        fvg_down_price = self.fvg.fvg_down[0]

        poc = self.vbp.poc if self.vbp.poc is not None else float('nan')
        poc_volume = self.vbp.poc_volume if hasattr(self.vbp, 'poc_volume') else float('nan')
        poc_delta = self.vbp.poc_delta if hasattr(self.vbp, 'poc_delta') else float('nan')
        va_low = self.vbp.va_low if self.vbp.va_low is not None else float('nan')
        va_high = self.vbp.va_high if self.vbp.va_high is not None else float('nan')

        row = {
            'timestamp': timestamp,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'rsi': rsi,
            'ema20': ema20,
            'ema50': ema50,
            'bos_bull': bos_bull,
            'bos_bear': bos_bear,
            'fvg_up_active': fvg_up_active,
            'fvg_down_active': fvg_down_active,
            'fvg_up_price': fvg_up_price,
            'fvg_down_price': fvg_down_price,
            'poc': poc,
            'poc_volume': poc_volume,
            'poc_delta': poc_delta,
            'va_low': va_low,
            'va_high': va_high
        }

        self.feature_rows.append(row)
    
def load_data_feed(file_path, cutoff_date):
    df = pd.read_csv(file_path, header=0, on_bad_lines='skip')
    
    df.columns = [
        "open_time","open","high","low","close","volume",
    ]

    df = df[["open_time", "open", "high", "low", "close", "volume"]] 
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    
    if cutoff_date is not None:
        df = df[df.index <= cutoff_date] 
    
    df.dropna(inplace=True)
    
    print("Dataframe Loaded")
    
    data = bt.feeds.PandasData(dataname=df)
    return data


def main():
    
    cerebro = bt.Cerebro(stdstats=False)
      
    data = load_data_feed("data/spot/all/SOLUSDT-4h-2020-08-2026-03.csv", "2026-04-01") #VARIABLE
    cerebro.adddata(data)
    
    cerebro.addstrategy(
        FeatureExtractedStrategy, 
        volume_level_file_path="data/spot/volume_levels/SOLUSDT-1m-2020-08-2026-03.parquet", #VARIABLE
        feature_save_file_path="data/feature_extracted/4H/SOLUSDT-4h-2020-08-2026-03-features.csv", #VARIABLE
    )
    print("Cerebro Loaded")

    cerebro.run(runonce=False)
    
    cerebro.plot()
    print("Cerebro Compleated")

if __name__ == "__main__":
    main()
    
    
    
    
    
"""

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
        if non_zero:
            volume_levels[ts] = non_zero
    

    
    vp = results[0].vbp
    vp.plot_volume_profile()


    #self.rsi = RSI(self.data, period=14)
    
    #self.ema20 = EMA(self.data, period=20)
    #self.ema50 = EMA(self.data, period=50)
    
    #self.structure = AdaptivePIP(self.data, atr_period=14, atr_multiplier=5)
    #self.bos = BreakOfStructure(pip_indicator=self.structure)
    
    #self.fvg = FVG(self.data, buffer=5)

    def save_values(self):
        timestamp = self.data.datetime.datetime(0)

        open_ = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        volume = self.data.volume[0]

        rsi = self.rsi.lines.rsi[0]
        ema20 = self.ema20.lines.ema[0]
        ema50 = self.ema50.lines.ema[0]

        bos_bull = self.bos.lines.bos_bull[0] if hasattr(self.bos.lines, 'bos_bull') else float('nan')
        bos_bear = self.bos.lines.bos_bear[0] if hasattr(self.bos.lines, 'bos_bear') else float('nan')

        fvg_up_event = self.fvg.fvg_up_event[0]
        fvg_down_event = self.fvg.fvg_down_event[0]
        fvg_up_active = self.fvg.fvg_up_active[0]
        fvg_down_active = self.fvg.fvg_down_active[0]
        fvg_up_price = self.fvg.fvg_up[0]
        fvg_down_price = self.fvg.fvg_down[0]

        poc = self.vbp.poc if self.vbp.poc is not None else float('nan')
        poc_volume = self.vbp.poc_volume if hasattr(self.vbp, 'poc_volume') else float('nan')
        poc_delta = self.vbp.poc_delta if hasattr(self.vbp, 'poc_delta') else float('nan')
        va_low = self.vbp.va_low if self.vbp.va_low is not None else float('nan')
        va_high = self.vbp.va_high if self.vbp.va_high is not None else float('nan')

        row = {
            'timestamp': timestamp,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'rsi': rsi,
            'ema20': ema20,
            'ema50': ema50,
            'bos_bull': bos_bull,
            'bos_bear': bos_bear,
            'fvg_up_event': fvg_up_event,
            'fvg_down_event': fvg_down_event,
            'fvg_up_active': fvg_up_active,
            'fvg_down_active': fvg_down_active,
            'fvg_up_price': fvg_up_price,
            'fvg_down_price': fvg_down_price,
            'poc': poc,
            'poc_volume': poc_volume,
            'poc_delta': poc_delta,
            'va_low': va_low,
            'va_high': va_high
        }

        self.feature_rows.append(row)

"""