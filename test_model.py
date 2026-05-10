import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os
import json

from dateutil.relativedelta import relativedelta
from datetime import datetime


class MLStrategy(bt.Strategy):

    params = dict(
        model_path=None,
        trade_start=None,
        trade_end=None,
        threshold_buy=0.6,
        feature_file_path=None,
    )

    def __init__(self):
        self.trade_log = []
        self.equity_curve = []

        package = joblib.load(self.p.model_path)
        self.model = package["model"]
        self.scalers = package["scalers"]
        self.features = package["features"]

        self.scaler_map = {
            "returns": ["log_return", "log_range"],
            "spatial": ["dist_to_poc", "dist_to_vah", "dist_to_val", "poc_delta"],
            "activity": ["volume", "poc_volume"],
            "trend": ["ema20", "ema50"],
        }

        feat_df = pd.read_csv(self.p.feature_file_path, parse_dates=["timestamp"])
        feat_df = feat_df.set_index("timestamp")

        feat_df["log_return"] = np.log(feat_df["close"] / feat_df["close"].shift(1))
        feat_df["log_range"]  = np.log(feat_df["high"] / feat_df["low"])

        feat_df["dist_to_poc"] = (feat_df["close"] - feat_df["poc"]) / feat_df["close"]
        feat_df["dist_to_vah"] = (feat_df["close"] - feat_df["va_high"]) / feat_df["close"]
        feat_df["dist_to_val"] = (feat_df["close"] - feat_df["va_low"]) / feat_df["close"]

        self.feat_df = feat_df

    def next(self):
        dt = self.data.datetime.datetime(0)

        if self.p.trade_start and dt < self.p.trade_start:
            return
        if self.p.trade_end and dt > self.p.trade_end:
            return

        self.equity_curve.append(self.broker.getvalue())

        if dt not in self.feat_df.index:
            return

        row = self.feat_df.loc[dt]

        if pd.isna(row.get("log_return")):
            return

        df = pd.DataFrame([row])

        for name, scaler in self.scalers.items():
            cols = self.scaler_map[name]
            if not all(c in df.columns for c in cols):
                continue
            df[cols] = scaler.transform(df[cols])

        X = df[self.features]

        prob = self.model.predict_proba(X)[0][1]

        if prob > self.p.threshold_buy:
            if not self.position:
                close = self.data.close[0]
                tp = close * 1.04 #VARIABLE
                sl = close * 0.96 #VARIABLE
                
                size = (self.broker.getcash() * 0.01) / close

                self.buy_bracket(size=size, limitprice=tp, stopprice=sl)

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
    path = os.path.join(base_path, "backtest_result_base_plus_rsi_and_ema.json") # VARIBALE
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
    return bt.feeds.PandasData(dataname=df)


def run_backtest(data_path, model_path, feature_file_path, start_date=None, end_date=None, warmup=50, timeframe=15):
    cerebro = bt.Cerebro()
    cerebro.addsizer(bt.sizers.FixedSize, stake=0)

    data = load_data(data_path, start_date=start_date, end_date=end_date, warmup_bars=warmup, timeframe_minutes=timeframe)
    cerebro.adddata(data)

    cerebro.addstrategy(
        MLStrategy,
        model_path=model_path,
        trade_start=pd.to_datetime(start_date) if start_date else None,
        trade_end=pd.to_datetime(end_date) if end_date else None,
        threshold_buy=0.55,
        feature_file_path=feature_file_path,
    )

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.0004)
    cerebro.broker.set_shortcash(False)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print("Starting Portfolio Value:", cerebro.broker.getvalue())
    result = cerebro.run(runonce=False)

    print("Final Portfolio Value:", cerebro.broker.getvalue())

    return result


if __name__ == "__main__":
    train_start = datetime(2021, 1, 1) # VARIBALE
    end_limit = datetime(2026, 4, 1) # VARIBALE

    while train_start + relativedelta(years=4) <= end_limit: # VARIBALE
        train_end = train_start + relativedelta(years=4) # VARIBALE
        test_start = train_end
        test_end = test_start + relativedelta(months=3)

        if test_end > end_limit:
            break

        folder = f"models/SOL_4H/4Years/{train_start.strftime('%Y-%m-%d')}_{train_end.strftime('%Y-%m-%d')}/" # VARIBALE

        result = run_backtest(
            data_path="data/spot/all/SOLUSDT-4h-2020-08-2026-03.csv", # VARIBALE
            model_path=f"{folder}base_plus_rsi_and_ema.pkl", # VARIBALE
            feature_file_path="data/feature_extracted/4H/SOLUSDT-4h-2020-08-2026-03-features.csv", # VARIBALE
            start_date=test_start.strftime("%Y-%m-%d"),
            end_date=test_end.strftime("%Y-%m-%d"),
            warmup=50,
            timeframe=240, #Var
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