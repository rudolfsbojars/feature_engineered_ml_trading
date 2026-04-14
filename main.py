import backtrader as bt
from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.smc import BreakOfStructure, FVG
from feature_algos.volume_profile import VolumeProfile
import pandas as pd
import os
from pathlib import Path
import glob

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
        
        self.fvg = FVG(self.data, buffer=5)
        self.vbp = VolumeProfile(
            self.data,
            bin_size=50,
            window_days=5,
            volume_file=volume_file
        )

        self.feature_rows = []
        
                
    def next(self):
        self.print_values()
        self.save_values()
        pass
        
        
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
        
    def print_values(self):

        open_ = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        close = self.data.close[0]
        volume = self.data.volume[0]

        print(
            f"{self.data.datetime.datetime(0)} | "
            f"O: {open_:.2f} | H: {high:.2f} | L: {low:.2f} | C: {close:.2f} | V: {volume:.6f}"
        )
    
        rsi = self.rsi.lines.rsi[0] 
        ema20 = self.ema20.lines.ema[0] 
        ema50 = self.ema50.lines.ema[0] 

        print(
            f"{self.data.datetime.datetime(0)} | RSI: {rsi:.2f} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f}")
    
        bos_bull = self.bos.lines.bos_bull[0]
        bos_bear = self.bos.lines.bos_bear[0]

        print(f"{self.data.datetime.datetime(0)} | BOS bull: {bos_bull} | BOS bear: {bos_bear}")
            
        fvg_up_event = self.fvg.fvg_up_event[0]
        fvg_down_event = self.fvg.fvg_down_event[0]

        fvg_up_active = self.fvg.fvg_up_active[0]
        fvg_down_active = self.fvg.fvg_down_active[0]

        fvg_up_price = self.fvg.fvg_up[0]
        fvg_down_price = self.fvg.fvg_down[0]
        
        print(
            f"{self.data.datetime.datetime(0)} | "
            f"FVG Up Event: {fvg_up_event} | FVG Down Event: {fvg_down_event} | "
            f"FVG Up Active: {fvg_up_active} | FVG Down Active: {fvg_down_active} | "
            f"FVG Up Price: {fvg_up_price} | FVG Down Price: {fvg_down_price}"
        )

        poc = self.vbp.poc
        poc_volume = self.vbp.poc_volume
        poc_delta = self.vbp.poc_delta
        va_low = self.vbp.va_low
        va_high = self.vbp.va_high

        if poc is not None and va_low is not None and va_high is not None:
            print(f"{self.data.datetime.datetime(0)} | POC: {poc} | POC Volume: {poc_volume} | VA: {va_low}-{va_high} | POC Delta: {poc_delta}")


def save_features_to_file(name, strategy):
    output_dir = "data/feature_extracted"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, name)

    df_features = pd.DataFrame(strategy.feature_rows)
    df_features = df_features.fillna(0)
    df_features.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    
    
def load_ohlcv_year(symbol: str, year: int, timeframe: str = '1m') -> pd.DataFrame:
    path = Path(f"data/raw/spot/monthly/klines/{symbol}/{timeframe}")
    all_months = [f"{symbol}-{timeframe}-{year}-{month:02d}.csv" for month in range(1, 13)]
    
    dfs = []
    for file in all_months:
        csv_path = path / file
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        
        df = pd.read_csv(csv_path, header=None)
        df.columns = [
            "open_time","open","high","low","close","volume_btc",
            "close_time","quote_volume_usd","num_trades",
            "taker_buy_vol_btc","taker_buy_vol_usd","ignore"
        ]
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df[['open','high','low','close','volume_btc']]
        df.rename(columns={'volume_btc':'volume'}, inplace=True)
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No OHLCV data found for {symbol} {year}")
    
    df_all = pd.concat(dfs)
    df_hourly = df_all.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_hourly.dropna(inplace=True)
    return df_hourly


def consolidate_vp_year(symbol: str, year: int, save_path: str = None) -> pd.DataFrame:
    vp_path_pattern = f"data/price_level_volumes/{symbol}/vp_{symbol}-1h-{year}-*.parquet"
    vp_files = sorted(glob.glob(vp_path_pattern))
    
    if not vp_files:
        raise ValueError(f"No VP files found for {symbol} {year}")

    dfs = []
    for file in vp_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    vp_all = pd.concat(dfs)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vp_all.to_parquet(save_path)
        print(f"Consolidated VP saved to {save_path}")
    
    return vp_all


def concat_files():
    symbol = "BTCUSDT"
    year = 2022
    concat_dir = Path("data/concat")
    concat_dir.mkdir(parents=True, exist_ok=True)

    df_hourly = load_ohlcv_year(symbol, year)
    ohlcv_file = concat_dir / f"{symbol}-1h-{year}-all.csv"
    df_hourly.to_csv(ohlcv_file)
    print(f"Hourly OHLCV saved to {ohlcv_file}")

    vp_file = concat_dir / f"vp_{symbol}-1h-{year}-all.parquet"
    vp_all = consolidate_vp_year(symbol, year, save_path=vp_file)

def main1():
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
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(Strategy, volume_file="data/price_level_volumes/BTCUSDT/vp_BTCUSDT-1h-2022-04.parquet")

    results = cerebro.run(runonce=False)
    strategy = results[0]
    
    save_features_to_file("BTCUSDT-1h-2022-04.csv", strategy=strategy)
    
    cerebro.plot()
    
    vp = results[0].vbp
    vp.plot_volume_profile()

def main():

    df = pd.read_csv("data/concat/BTCUSDT-1h-2022-all.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    """
    df_hourly = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_hourly.dropna(inplace=True)
    """

    data = bt.feeds.PandasData(dataname=df)
    
    
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(Strategy, volume_file="data/concat/vp_BTCUSDT-1h-2022-all.parquet")

    results = cerebro.run(runonce=False)
    strategy = results[0]
    
    save_features_to_file("BTCUSDT-1h-2022-04.csv", strategy=strategy)
    
    cerebro.plot()
    
    vp = results[0].vbp
    vp.plot_volume_profile()
    
    
# each stage runs after another data and so on, create true false switche to toggle somthign on
# Create import data, create parquete data, read parquests, adatta, make the indicators, 
# pre proccess normalize, label data, 
# train model, test model. split itot o data, and train,

#normalize imediatly after findign indicator, have anotehr flow
    
        
if __name__ == '__main__':
    
    main1()
    
    
# z-score normalization and then save again an then feed it to the model