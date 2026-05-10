import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
 
from feature_algos.rsi import RSI
from feature_algos.pip import AdaptivePIP
from feature_algos.ema import EMA
from feature_algos.smc import BreakOfStructure, FVG
from feature_algos.volume_profile import VolumeProfile, LookbackWindowMode
 
# ── config ────────────────────────────────────────────────────────────
DATA_PATH          = "data/spot/all/BTCUSDT-4h-2017-08-2026-03.csv"
VOLUME_LEVELS_PATH = "data/spot/volume_levels/BTCUSDT-1m-2017-08-2026-03.parquet"
START_DATE = "2025-01-01"
END_DATE   = "2025-12-31 23:59:59"
WINDOW             = 80      # how many candles to show at once
PAUSE              = 0.001   # seconds between redraws (lower = faster)
START_AT           = 60      # skip first N bars (warm-up)
# ──────────────────────────────────────────────────────────────────────
 
 
class LivePlotStrategy(bt.Strategy):
 
    def __init__(self, volume_level_file_path=None):
        self.rsi   = RSI(self.data, period=14, upper=70, lower=30)
        self.ema20 = EMA(self.data, period=20)
        self.ema50 = EMA(self.data, period=50)
        self.pip   = AdaptivePIP(self.data, atr_period=14, atr_multiplier=5)
        self.bos   = BreakOfStructure(self.data, lookback=50, pip_indicator=self.pip)
        self.fvg   = FVG(self.data)
        self.vbp   = VolumeProfile(
            self.data,
            level_count_per_range=50,
            lookback_window_mode=LookbackWindowMode.WEEK,
            volume_level_file_path=volume_level_file_path,
            volume_area=0.7,
        )
 
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(16, 9),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True
        )
        self.fig.tight_layout(pad=2)
 
    def next(self):
        bar = len(self) - 1
        if bar < START_AT:
            return
 
        # ── slice the visible window ──────────
        size   = min(bar + 1, WINDOW)
        xs     = list(range(bar - size + 1, bar + 1))
        
        dates = [
            self.data.datetime.datetime(-i)
            for i in range(size - 1, -1, -1)
        ]
 
        closes = [self.data.close[-i]  for i in range(size - 1, -1, -1)]
        opens  = [self.data.open[-i]   for i in range(size - 1, -1, -1)]
        highs  = [self.data.high[-i]   for i in range(size - 1, -1, -1)]
        lows   = [self.data.low[-i]    for i in range(size - 1, -1, -1)]
 
        ema20  = [self.ema20.lines.ema[-i] for i in range(size - 1, -1, -1)]
        ema50  = [self.ema50.lines.ema[-i] for i in range(size - 1, -1, -1)]
        rsi    = [self.rsi.lines.rsi[-i]   for i in range(size - 1, -1, -1)]
 
        # ── clear and redraw ──────────────────
        self.ax1.cla()
        self.ax2.cla()
 
        # candlesticks
        w = 0.4
        for i, x in enumerate(xs):
            color = "green" if closes[i] >= opens[i] else "red"
            lo    = min(opens[i], closes[i])
            hi    = max(opens[i], closes[i])
            self.ax1.add_patch(plt.Rectangle(
                (x - w, lo), 2 * w, max(hi - lo, 1e-9),
                color=color, zorder=3
            ))
            self.ax1.plot([x, x], [lows[i],  lo], color="gray", lw=0.8, zorder=2)
            self.ax1.plot([x, x], [hi, highs[i]], color="gray", lw=0.8, zorder=2)
 
        # EMA
        self.ax1.plot(xs, ema20, color="orange", lw=1.2, label="EMA 20")
        self.ax1.plot(xs, ema50, color="purple", lw=1.2, linestyle="--", label="EMA 50")
 
        # PIPs  (scan window for confirmed peaks/valleys)
        pk_xs, pk_ys, vl_xs, vl_ys = [], [], [], []
        for i in range(-size + 1, 1):
            abs_i = bar + i
            try:
                pv = self.pip.lines.peak[i]
                if not np.isnan(pv):
                    pk_xs.append(abs_i); pk_ys.append(pv)
            except Exception:
                pass
            try:
                vv = self.pip.lines.valley[i]
                if not np.isnan(vv):
                    vl_xs.append(abs_i); vl_ys.append(vv)
            except Exception:
                pass
 
        # structure line through all confirmed pips
        all_pips = sorted(
            [(x, y) for x, y in zip(pk_xs, pk_ys)] +
            [(x, y) for x, y in zip(vl_xs, vl_ys)],
            key=lambda t: t[0]
        )
        if len(all_pips) >= 2:
            sx = [p[0] for p in all_pips]
            sy = [p[1] for p in all_pips]
            self.ax1.plot(sx, sy, color="steelblue", lw=1.0, alpha=0.7, label="PIP structure")
 
        self.ax1.scatter(pk_xs, pk_ys, marker="^", s=60, color="lime",   zorder=5, label="Peak")
        self.ax1.scatter(vl_xs, vl_ys, marker="v", s=60, color="tomato", zorder=5, label="Valley")
 
        # BOS
        bos_bull = self.bos.lines.bos_bull[0]
        bos_bear = self.bos.lines.bos_bear[0]
        if bos_bull == 1.0:
            self.ax1.annotate("↑BOS", xy=(bar, self.data.close[0]),
                              color="gold", fontsize=8, fontweight="bold",
                              xytext=(bar, self.data.close[0] * 1.003),
                              arrowprops=dict(arrowstyle="-", color="gold", lw=0.8))
        if bos_bear == 1.0:
            self.ax1.annotate("↓BOS", xy=(bar, self.data.close[0]),
                              color="salmon", fontsize=8, fontweight="bold",
                              xytext=(bar, self.data.close[0] * 0.997),
                              arrowprops=dict(arrowstyle="-", color="salmon", lw=0.8))
 
        # FVG zones
        for lo, hi in self.fvg._bull_zones:
            self.ax1.axhspan(lo, hi, alpha=0.15, color="green", zorder=1)
        for lo, hi in self.fvg._bear_zones:
            self.ax1.axhspan(lo, hi, alpha=0.15, color="red",   zorder=1)
 
        # Volume profile
        if self.vbp.va_low is not None and self.vbp.va_high is not None:
            self.ax1.axhspan(self.vbp.va_low, self.vbp.va_high,
                             alpha=0.08, color="cyan", zorder=1)
            self.ax1.axhline(self.vbp.va_high, color="cyan", lw=0.8,
                             linestyle="--", alpha=0.6, label=f"VAH {self.vbp.va_high:.2f}")
            self.ax1.axhline(self.vbp.va_low,  color="cyan", lw=0.8,
                             linestyle="--", alpha=0.6, label=f"VAL {self.vbp.va_low:.2f}")
        if self.vbp.poc is not None:
            self.ax1.axhline(self.vbp.poc, color="orangered", lw=1.2,
                             linestyle="-.", alpha=0.9, label=f"POC {self.vbp.poc:.2f}")
 
        self.ax1.set_title(f"Bar {bar}  |  close {self.data.close[0]:.2f}", fontsize=9)
        self.ax1.legend(loc="upper left", fontsize=7, ncol=4)
        self.ax1.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
 
        # RSI
        self.ax2.plot(xs, rsi, color="violet", lw=1.2)
        self.ax2.axhline(70, color="red",   lw=0.7, linestyle="--")
        self.ax2.axhline(30, color="green", lw=0.7, linestyle="--")
        self.ax2.axhline(50, color="gray",  lw=0.4, linestyle=":")
        self.ax2.set_ylim(0, 100)
        self.ax2.set_ylabel("RSI", fontsize=8)
        self.ax2.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
        self.ax2.set_xticklabels(
            [d.strftime("%Y-%m-%d") for d in dates[::max(1, len(dates)//8)]],
            rotation=45,
            ha="right",
            fontsize=8
        )
 
        plt.pause(PAUSE)
 
    def stop(self):
        plt.ioff()
        plt.show()
 
 
def load_data(path, start=None, end=None):
    df = pd.read_csv(path, header=0, on_bad_lines="skip")
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")

    if start:
        df = df[df.index >= start]

    if end:
        df = df[df.index <= end]
    df.dropna(inplace=True)
    return bt.feeds.PandasData(dataname=df)
 
 
def main():
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(load_data(
        DATA_PATH,
        start=START_DATE,
        end=END_DATE,
    ))
    cerebro.addstrategy(
        LivePlotStrategy,
        volume_level_file_path=VOLUME_LEVELS_PATH,
    )
    cerebro.run(runonce=False)
 
 
if __name__ == "__main__":
    main()