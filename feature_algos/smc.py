import backtrader as bt
import numpy as np

class BreakOfStructure(bt.Indicator):

    lines = ('bos_bull', 'bos_bear')
    plotlines = dict(
        bos=dict(marker='o', markersize=6.0, color='orange', ls='', _name='BOS')
    )

    params = dict(
        lookback=50,
    )

    def __init__(self, pip_indicator):

        self.pips = pip_indicator

    def next(self):
        close = self.data.close[0]

        last_peak = None
        last_valley = None

        for i in range(-self.p.lookback, 0):
            if i >= -len(self.pips.peak):
                peak = self.pips.peak[i]
                valley = self.pips.valley[i]

                if not np.isnan(peak):
                    last_peak = peak
                if not np.isnan(valley):
                    last_valley = valley

        self.lines.bos_bull[0] = 0.0
        self.lines.bos_bear[0] = 0.0

        if last_peak is not None and close > last_peak:
            self.lines.bos_bull[0] = 1.0

        elif last_valley is not None and close < last_valley:
            self.lines.bos_bear[0] = 1.0
            
            
class FVG(bt.Indicator):
    lines = (
        'fvg_up', 'fvg_down',
        'fvg_up_event', 'fvg_down_event',
        'fvg_up_active', 'fvg_down_active'
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(
        fvg_up=dict(color='lime', ls='--', linewidth=2, _name='FVG Up'),
        fvg_down=dict(color='red', ls='--', linewidth=2, _name='FVG Down')
    )

    params = dict(
        buffer=0.001,
    )

    def __init__(self):
        self._bull_zones = []
        self._bear_zones = []

    def next(self):
        if len(self.data.close) < 3:
            return

        h0, h1, h2 = self.data.high[-3], self.data.high[-2], self.data.high[-1]
        l0, l1, l2 = self.data.low[-3], self.data.low[-2], self.data.low[-1]

        self.lines.fvg_up_event[0] = 0.0
        self.lines.fvg_down_event[0] = 0.0

        if l1 > h0 and l1 > h2 - self.params.buffer:
            self._bull_zones.append((h0, l1))
            self.lines.fvg_up_event[0] = 1.0

        if h1 < l0  and h1 < l2 + self.params.buffer:
            print("opa")
            self._bear_zones.append((h1, l0))
            self.lines.fvg_down_event[0] = 1.0

        for zone in self._bull_zones[:]:
            if self.data.low[0] <= zone[0]:
                self._bull_zones.remove(zone)

        for zone in self._bear_zones[:]:
            if self.data.high[0] >= zone[1]:
                self._bear_zones.remove(zone)

        self.lines.fvg_up[0] = float('nan')
        self.lines.fvg_down[0] = float('nan')
        self.lines.fvg_up_active[0] = 0.0
        self.lines.fvg_down_active[0] = 0.0

        if self._bull_zones:
            level = min([z[0] for z in self._bull_zones])
            self.lines.fvg_up[0] = level
            self.lines.fvg_up_active[0] = 1.0

        if self._bear_zones:
            level = max([z[1] for z in self._bear_zones])
            self.lines.fvg_down[0] = level
            self.lines.fvg_down_active[0] = 1.0
            print("yeeeee")