import backtrader as bt


class EMA(bt.Indicator):
    lines = ('ema',)

    params = (
        ('period', 20),
        ('color', 'orange'),
    )

    plotinfo = dict(subplot=False, plotname='EMA')

    plotlines = dict(
        ema=dict(_name='EMA', color='orange', ls='-')
    )

    def __init__(self):
        self.multiplier = 2.0 / (self.p.period + 1)
        self.ema_value = 0.0
        self.initialized = False

    def next(self):
        if len(self) < self.p.period:
            return

        if len(self) == self.p.period:

            closes = [self.data.close[-i] for i in range(self.p.period)]
            self.ema_value = sum(closes) / self.p.period
            self.initialized = True

        elif self.initialized:
            self.ema_value = (self.data.close[0] - self.ema_value) * self.multiplier + self.ema_value

        self.lines.ema[0] = self.ema_value