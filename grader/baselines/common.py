"""Small interface helpers used by isolated baseline candidates."""
import numpy as np
import pandas as pd


class Baseline:
    markets = ['ETH']
    resolution = '1h'
    max_horizon_bars = 1
    lookback_bars = 1
    budget_tier = 'fast'
    uses_dev_lake = False

    def fit(self, train, ctx):
        pass

    def positions(self, history, decision_times, transform):
        frames = []
        for market in self.markets:
            data = history.datasets[market].frame
            if not np.isfinite(data.close).all() or data.close.le(0).any():
                raise ValueError('Baseline received unknown/nonpositive prices')
            index = pd.DatetimeIndex(pd.to_datetime(data.available_at_ms, unit='ms', utc=True))
            values = pd.Series(transform(data), index=index).reindex(decision_times)
            if values.isna().any():
                raise ValueError('Insufficient completed baseline history')
            frames.append(pd.DataFrame({'timestamp': decision_times, 'market': market,
                                        'target_position': values.to_numpy(dtype=float)}))
        return pd.concat(frames, ignore_index=True)
