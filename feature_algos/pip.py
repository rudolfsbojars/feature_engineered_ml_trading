import backtrader as bt
import numpy as np

class AdaptivePIP(bt.Indicator):

    lines = ('peak', 'valley', 'structure')
    params = (
        ("atr_period", 14),
        ("atr_multiplier", 10),
    )

    plotinfo = dict(subplot=False)
    plotlines = dict(
        peak=dict(marker='^', markersize=8.0, color='green', ls='', _name='Peak'),
        valley=dict(marker='v', markersize=8.0, color='red', ls='', _name='Valley'),
        structure=dict(color='blue', ls='-', linewidth=2, _name='Structure'),
    )
    
    def __init__(self):

        self.addminperiod(self.params.atr_period)

    def next(self):
        
        for i in range(-self.data.buflen(), 0):
            self.lines.structure[i] = float('nan')
            self.lines.peak[i] = float('nan')
            self.lines.valley[i] = float('nan')

        data = np.array(self.data.close.get(size=self.data.buflen()))

        pips = self.get_perceptually_important_points(data)
        
        for i in range(len(pips)):
            bt_index = pips[i][1] - self.data.buflen() + 1
            
            
            #Visualize Maxmimums and minimums
            match pips[i][2]:
                case "peak":
                    self.lines.peak[bt_index] = pips[i][0]
                case "valley":
                    self.lines.valley[bt_index] = pips[i][0]
                     
            #Visualize Structure
            if i < len(pips) - 1:
                run = pips[i+1][1] - pips[i][1]
                
                if run > 0:
                    rise = pips[i+1][0] - pips[i][0]
                    slope = rise / run
                    
                    for step in range(run + 1):
                        self.lines.structure[bt_index + step] = pips[i][0] + (slope * step)  
                           
    
    def get_perceptually_important_points(self, data):
        data = list(data)
        n = len(data)
        if n < 3:
            return [(data[i], i, "none") for i in range(n)]

        def segment_atr(start):
            lo = max(0, start - self.params.atr_period)
            seg = data[lo:start + 1]
            if len(seg) < 2:
                return 1e-8
            ranges = [abs(seg[i] - seg[i - 1]) for i in range(1, len(seg))]
            window = ranges[-self.params.atr_period:] if len(ranges) >= self.params.atr_period else ranges
            return max(float(np.mean(window)), 1e-8)

        pips_indices = [0, n - 1]
        pips_set = set(pips_indices)
        directions = {}

        max_iter = n
        for _ in range(max_iter):
            pips_indices.sort()
            added_any = False

            for i in range(len(pips_indices) - 1):
                start_idx = pips_indices[i]
                end_idx = pips_indices[i + 1]
                if end_idx - start_idx < 2:
                    continue

                local_atr = segment_atr(start_idx)
                threshold = self.params.atr_multiplier * local_atr

                x1, x2 = start_idx, end_idx
                y1, y2 = data[x1], data[x2]
                dx, dy = x2 - x1, y2 - y1
                denom = np.sqrt(dx**2 + dy**2)

                best_dist = -1
                best_idx = -1
                best_price_dist = 0.0

                for j in range(1, end_idx - start_idx):
                    curr_idx = x1 + j
                    if denom > 0:
                        num = abs(dy * curr_idx - dx * data[curr_idx] + x2 * y1 - y2 * x1)
                        euc_dist = num / denom
                    else:
                        euc_dist = abs(data[curr_idx] - y1)

                    if euc_dist > best_dist:
                        best_dist = euc_dist
                        best_idx = curr_idx
                        interp = y1 + (dy / dx) * (curr_idx - x1) if dx != 0 else y1
                        best_price_dist = data[curr_idx] - interp

                if best_idx != -1 and abs(best_price_dist) >= threshold and best_idx not in pips_set:
                    pips_indices.append(best_idx)
                    pips_set.add(best_idx)
                    directions[best_idx] = "peak" if best_price_dist > 0 else "valley"
                    added_any = True

            if not added_any:
                break

        pips_indices.sort()

        if len(pips_indices) >= 2:
            i0, i1 = pips_indices[0], pips_indices[1]
            directions[i0] = "valley" if data[i0] < data[i1] else "peak"
            ie, ip = pips_indices[-1], pips_indices[-2]
            directions[ie] = "valley" if data[ie] < data[ip] else "peak"

        return [(data[idx], idx, directions.get(idx, "none")) for idx in pips_indices]
        
    
    """
    
    Best out of Directional Change, Rolling window (fractals)
    
    Modified Ramer-Douglas-Peucker algorithm
    
    The function builds a self-terminating, volatility-aware skeleton of the most geometrically 
    significant turning points in a price series. Here's what it keeps in mind and what each part is 
    doing:

    The core idea — divide and conquer
    It starts with just two points: the very first and last bar. Then it keeps asking: 
    "between any two known points, is there a bar that deviates from the straight line connecting 
    them by more than the noise floor?" If yes, that bar becomes a new PIP. It repeats until no 
    segment has a point that clears the threshold. The result is the minimum set of points needed 
    to faithfully describe the shape of the move.
    
    This mirrors exactly how a human draws a chart by hand — you place your pen at the start, jump 
    to the most obvious turning point, then fill in secondary pivots, and stop when the remaining 
    wiggles feel like noise.

    What it keeps in mind
    1. Volatility is local, not global. The ATR threshold is recalculated for each segment independently. 
        A segment covering a low-volatility consolidation gets a tight threshold — small moves count. 
        A segment covering a high-volatility impulse gets a wide threshold — only large swings qualify. 
        This means the algorithm adapts its sensitivity to the current market regime rather than applying 
        one global filter that over-fires in quiet periods and under-fires in explosive ones.
    2. Geometric distance, not vertical distance. The deviation of a candidate point from the line 
        connecting two PIPs is measured as true perpendicular Euclidean distance, not just how far above or 
        below the interpolated price it sits. This matters on steep diagonal segments — a point slightly off 
        a sharp trend line reads as insignificant on vertical distance but significant on geometric distance. 
        The Euclidean version is what the human eye actually responds to.
    3. Price significance, not just geometry. After finding the geometrically furthest point in a 
        segment, it still checks whether the vertical price deviation exceeds atr_multiplier x
        local_atr. Geometry alone would flag statistically insignificant wiggles on near-flat segments. 
        The ATR gate ensures every accepted PIP represents a move that's meaningful relative to recent 
        volatility — the threshold is expressed in price terms, not pixel terms.
    4. The data drives the count. There's no n_pips parameter. The function keeps running until 
        no new point clears the threshold anywhere. On a trending, structured market it will naturally produce 
        more pivots. On a flat, noisy series it will produce fewer. This self-terminating behaviour means 
        the output reflects the actual complexity of the price action rather than an arbitrary pivot count you 
        tuned by hand.
    5. The hierarchy is correct. Because it processes the highest-significance point in each segment first,
        the output implicitly encodes pivot importance. The first PIPs added are the major structure — 
        the most geometrically important turns. Later PIPs are secondary. If you truncate the process early 
        (by raising atr_multiplier), you're left with only the highest-timeframe structure, which is exactly 
        the behaviour you want when zooming out.

    RIGID, standart pip finder with defined nr of points, not great for when markets go sideways and the strucutre really isnt there
    
    class PerceptuallyImportantPoints(bt.Indicator):

    lines = ('peak', 'valley', 'structure')
    params = (
        ('lookback', 200), 
    )

    plotinfo = dict(subplot=False)
    plotlines = dict(
        peak=dict(marker='v', markersize=8.0, color='red', ls=''),
        valley=dict(marker='^', markersize=8.0, color='green', ls=''),
        structure=dict(color='blue', ls='-', linewidth=2, _name='Structure'),
    )
    
    def __init__(self):

        self.addminperiod(self.params.lookback)

    def next(self):
        for i in range(-self.params.lookback, 0):
            self.lines.structure[i] = float('nan')
            self.lines.peak[i] = float('nan')
            self.lines.valley[i] = float('nan')

        data = np.array(self.data.close.get(size=self.params.lookback))

        pips = self.get_perceptually_important_points(data, 20)
        print(pips)
        
        for i in range(len(pips)):
            bt_index = pips[i][1] - self.params.lookback + 1
            self.lines.valley[bt_index] = pips[i][0]

        
        
    def get_perceptually_important_points(self, data, n_pips):
        pips_indices = [0, len(data) - 1]
        directions = {0: "none", len(data) - 1: "none"}
        
        while len(pips_indices) < n_pips:
            max_dist = -1
            best_idx = -1
            direction = "none"
            
            pips_indices.sort()
            
            for i in range(len(pips_indices) - 1):
                start_idx = pips_indices[i]
                end_idx = pips_indices[i+1]
                
                y1, y2 = data[start_idx], data[end_idx]
                x1, x2 = start_idx, end_idx
                
                segment = data[start_idx : end_idx + 1]
                
                for j in range(len(segment)):
                    curr_idx = start_idx + j
                    if x2 - x1 == 0: continue
                    slope = (y2 - y1) / (x2 - x1)
                    expected_y = y1 + slope * (curr_idx - x1)
                    dist = data[curr_idx] - expected_y
                    
                    if abs(dist) > max_dist:
                        max_dist = abs(dist)
                        best_idx = curr_idx
                        direction = "peak" if dist > 0 else "valley"
                        
            if best_idx != -1 and best_idx not in pips_indices:
                pips_indices.append(best_idx)
                directions[best_idx] = direction
            else:
                break
                
        pips_indices.sort()
        return [(data[idx], idx, directions[idx]) for idx in pips_indices]
    """