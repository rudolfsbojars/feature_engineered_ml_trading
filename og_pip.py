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
        
    