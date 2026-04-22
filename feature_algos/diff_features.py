import backtrader as bt
import numpy as np


class DiffFeatures(bt.Indicator):
    lines = ('dist_to_poc', 'dist_to_vah', 'dist_to_val')
    plotinfo = dict(plot=False)

    def __init__(self, volume_profile):
        self.vbp = volume_profile

    def next(self):
        close = self.data.close[0]

        poc = self.vbp.poc if self.vbp.poc is not None else np.nan
        vah = self.vbp.va_high if self.vbp.va_high is not None else np.nan
        val = self.vbp.va_low if self.vbp.va_low is not None else np.nan

        self.lines.dist_to_poc[0] = close - poc
        self.lines.dist_to_vah[0] = close - vah
        self.lines.dist_to_val[0] = close - val