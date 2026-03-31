import backtrader as bt

class VWAP(bt.Indicator):
    lines = ('vwap',)

    params = (
        ('period', 14),
    )

    plotinfo = dict(subplot=False, plotname='VWAP')

    plotlines = dict(
        vwap=dict(_name='VWAP', color='purple', ls='-')
    )

    def __init__(self):
        self.cum_tp_vol = 0.0
        self.cum_vol = 0.0

    def next(self):
        if len(self) < self.p.period:
            return

        cum_tp_vol = 0.0
        cum_vol = 0.0

        for i in range(-self.p.period + 1, 1):
            high  = self.data.high[i]
            low   = self.data.low[i]
            close = self.data.close[i]
            vol   = self.data.volume[i]

            typical_price = (high + low + close) / 3.0
            cum_tp_vol += typical_price * vol
            cum_vol    += vol

        if cum_vol > 0:
            self.lines.vwap[0] = cum_tp_vol / cum_vol
        else:
            self.lines.vwap[0] = self.data.close[0]