import backtrader as bt

class RSI(bt.Indicator):
    lines = ('rsi', 'upper_band', 'lower_band')

    params = (
        ('period', 14),
        ('upper', 70),
        ('lower', 30),
    )
    
    plotinfo = dict(subplot=True, plotname='Manual RSI')
    
    plotlines = dict(
        rsi=dict(_name='RSI', color='purple'),
        upper_band=dict(_name='', color='blue', ls='--'),
        lower_band=dict(_name='', color='red', ls='--')
    )

    def __init__(self):
        self.lines.upper_band = bt.Max(0, self.p.upper)
        self.lines.lower_band = bt.Max(0, self.p.lower)
        
        self.avg_gain = 0.0
        self.avg_loss = 0.0

    def next(self):
        if len(self) < 2:
            return

        diff = self.data.close[0] - self.data.close[-1]
        gain = max(diff, 0)
        loss = max(-diff, 0)

        if len(self) == self.params.period + 1:
            gains = []
            losses = []
            for i in range(-self.params.period + 1, 1):
                d = self.data.close[i] - self.data.close[i-1]
                gains.append(max(d, 0))
                losses.append(max(-d, 0))
            
            self.avg_gain = sum(gains) / self.params.period
            self.avg_loss = sum(losses) / self.params.period

        elif len(self) > self.params.period + 1:
            self.avg_gain = (self.avg_gain * (self.params.period - 1) + gain) / self.params.period
            self.avg_loss = (self.avg_loss * (self.params.period - 1) + loss) / self.params.period

        if self.avg_loss == 0:
            self.lines.rsi[0] = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.lines.rsi[0] = 100.0 - (100.0 / (1.0 + rs))