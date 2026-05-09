import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os
import json

from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.smc import BreakOfStructure, FVG
from feature_algos.volume_profile import VolumeProfile, LookbackWindowMode
from feature_algos.diff_features import DiffFeatures

from dateutil.relativedelta import relativedelta
from datetime import datetime



class MLStrategy(bt.Strategy):

    params = dict(
        model_path=None,
        trade_start=None,
        trade_end=None,
        threshold_buy=0.6,
        threshold_sell=0.4,
        volume_level_file_path=None,
    )

    def __init__(self):
        self.trade_log = []
        self.equity_curve = []
        
        package = joblib.load(self.p.model_path)

        self.model = package["model"]
        self.scalers = package["scalers"]
        self.features = package["features"]

        self.rsi = RSI(self.data, period=14, upper=70, lower=30)
        self.ema20 = EMA(self.data, period=20)
        self.ema50 = EMA(self.data, period=50)
        self.pip = AdaptivePIP(self.data, atr_period=14, atr_multiplier=5)
        self.pip.plotinfo.plot = False
        self.bos = BreakOfStructure(self.data, lookback=50, pip_indicator=self.pip)
        self.fvg = FVG(self.data)
        self.vbp = VolumeProfile(
            self.data,
            level_count_per_range=50,
            lookback_window_mode=LookbackWindowMode.WEEK,
            volume_level_file_path=self.p.volume_level_file_path,
            volume_area=0.7,
        )
        
        self.diff = DiffFeatures(self.data, volume_profile=self.vbp)

        self.prev_close = None
        
        
                
        self.scaler_map = {
            "returns": ["log_return", "log_range"],
            "spatial": ["dist_to_poc", "dist_to_vah", "dist_to_val", "poc_delta"],
            "activity": ["volume", "poc_volume"],
            "trend": ["ema20", "ema50"],
        }
    

    def next(self):
        dt = self.data.datetime.datetime(0)
        
        if self.p.trade_start and dt < self.p.trade_start:
            return

        if self.p.trade_end and dt > self.p.trade_end:
            return
        
        print("Candle at: ", dt)
        
        self.equity_curve.append(self.broker.getvalue())


        close = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]
        volume = self.data.volume[0]

        if self.prev_close is None:
            self.prev_close = close
            return

        log_return = np.log(close / self.prev_close)
        log_range = np.log(high / low)
        
        row = {
            "log_return": log_return,
            "log_range": log_range,
            "volume": volume,
            "rsi": self.rsi.lines.rsi[0],
            "ema20": self.ema20.lines.ema[0],
            "ema50": self.ema50.lines.ema[0],
            "bos_bull": getattr(self.bos.lines, "bos_bull", [np.nan])[0],
            "bos_bear": getattr(self.bos.lines, "bos_bear", [np.nan])[0],
            "fvg_up_active": self.fvg.fvg_up_active[0],
            "fvg_down_active": self.fvg.fvg_down_active[0],
            "poc": self.vbp.poc if self.vbp.poc is not None else np.nan,
            "va_high": self.vbp.va_high if self.vbp.va_high is not None else np.nan,
            "va_low": self.vbp.va_low if self.vbp.va_low is not None else np.nan,
            "poc_delta": getattr(self.vbp, "poc_delta", np.nan),
            "poc_volume": getattr(self.vbp, "poc_volume", np.nan),
            "dist_to_poc": self.diff.lines.dist_to_poc[0],
            "dist_to_vah": self.diff.lines.dist_to_vah[0],
            "dist_to_val": self.diff.lines.dist_to_val[0],
        }

        self.prev_close = close

        df = pd.DataFrame([row])

        for name, scaler in self.scalers.items():
            cols = self.scaler_map[name]

            if not all(c in df.columns for c in cols):
                continue

            df[cols] = scaler.transform(df[cols])

        X = df[self.features]


        prob = self.model.predict_proba(X)[0][1]
        
        print(f"prob: {prob:.4f}, position: {bool(self.position)}")

        if prob > self.p.threshold_buy:
            if not self.position:
                close = self.data.close[0]
                tp = close * 1.01
                sl = close * 0.99
                self.buy_bracket(
                    limitprice=tp,
                    stopprice=sl,
                )
                
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
                "pnl": trade.pnl,
                "pnlcomm": trade.pnlcomm,
                "size": trade.size,
            })
            
def monte_carlo(equity_curve, n=1000):
    returns = np.diff(equity_curve) / equity_curve[:-1]

    results = []

    for _ in range(n):
        sampled = np.random.choice(returns, size=len(returns), replace=True)
        curve = [10000]

        for r in sampled:
            curve.append(curve[-1] * (1 + r))

        results.append(curve[-1])

    return {
        "mc_mean": float(np.mean(results)),
        "mc_std": float(np.std(results)),
        "mc_min": float(np.min(results)),
        "mc_max": float(np.max(results)),
    }
    
def save_results(base_path, data):
    os.makedirs(base_path, exist_ok=True)

    path = os.path.join(base_path, "backtest_result.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=str)

    return path
                
                
def load_data(file_path, start_date=None, end_date=None, warmup_bars=0, timeframe_minutes=15):

    df = pd.read_csv(file_path)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("open_time").set_index("open_time")

    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if start_date is not None:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)

        warmup_ms = warmup_bars * timeframe_minutes * 60 * 1000
        actual_start_ts = start_ts - warmup_ms

        df = df[df.index.astype("int64") >= actual_start_ts]

    if end_date is not None:
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        df = df[df.index.astype("int64") <= end_ts]
        
    print(len(df))
    print(df.head())
    print(df.index[:5])

    return bt.feeds.PandasData(dataname=df)


def run_backtest(data_path, model_path, start_date=None, end_date=None, warmup=50, timeframe=15, volume_level_file_path=None):
    cerebro = bt.Cerebro()

    data = load_data(data_path, start_date=start_date, end_date=end_date, warmup_bars=warmup, timeframe_minutes=timeframe)
    cerebro.adddata(data)

    cerebro.addstrategy(
        MLStrategy,
        model_path=model_path,
        trade_start=pd.to_datetime(start_date) if start_date else None,
        trade_end=pd.to_datetime(end_date) if end_date else None,
        threshold_buy=0.55,
        threshold_sell=0.4,
        volume_level_file_path=volume_level_file_path
    )

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.0004)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print("Starting Portfolio Value:", cerebro.broker.getvalue())

    result = cerebro.run(runonce=False)
    
    cerebro.plot()
    
    print("Final Portfolio Value:", cerebro.broker.getvalue())

    return result


if __name__ == "__main__":

    train_start = datetime(2018, 1, 1)
    end_limit = datetime(2026, 4, 1)

    while train_start + relativedelta(years=1) <= end_limit:
        train_end = train_start + relativedelta(years=1)
        test_start = train_end
        test_end = test_start + relativedelta(months=3)

        if test_end > end_limit:
            break

        folder = f"models/BTC_15M/{train_start.strftime('%Y-%m-%d')}_{train_end.strftime('%Y-%m-%d')}/"

        result = run_backtest(
            data_path="data/spot/all/BTCUSDT-15m-2017-08-2026-03.csv",
            model_path=f"{folder}full.pkl",
            volume_level_file_path="data/spot/volume_levels/BTCUSDT-1m-2017-08-2026-03.parquet",
            start_date=test_start.strftime("%Y-%m-%d"),
            end_date=test_end.strftime("%Y-%m-%d"),
            warmup=50,
            timeframe=15,
        )

        strat = result[0]

        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        mc = monte_carlo(strat.equity_curve)

        final_report = {
            "sharpe": sharpe,
            "drawdown": drawdown,
            "trades": trades,
            "returns": returns,
            "monte_carlo": mc,
            "total_trades": len(strat.trade_log),
            "trade_log": strat.trade_log,
        }

        save_results(folder, final_report)

        train_start += relativedelta(months=3)