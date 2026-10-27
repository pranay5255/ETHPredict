"""Small interface helpers used by isolated baseline candidates."""
import numpy as np
import pandas as pd
from ..feature_packets import FeaturePacket, feature_noise, NOISE_NAMES


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


class MomentumBaseline(Baseline):
    engineered_features = ['returns']
    hours = 1

    def __init__(self):
        from ..data import RESOLUTIONS
        self.bars = self.hours * 3_600_000 // RESOLUTIONS[self.resolution]
        self.lookback_bars = self.bars + 1

    def fit(self, train, ctx):
        self.seed = ctx.seed

    def feature_packet(self, history, decision_times):
        frames = []
        for market in self.markets:
            source = history.datasets[market].frame
            index = pd.DatetimeIndex(pd.to_datetime(source.available_at_ms,unit='ms',utc=True))
            momentum = pd.Series((source.close/source.close.shift(self.bars)-1).to_numpy(),index=index).reindex(decision_times)
            noise = feature_noise(pd.DatetimeIndex(decision_times).as_unit('ms').asi8,seed=self.seed,market=market)
            frames.append(pd.DataFrame({'timestamp':decision_times,'market':market,'momentum':momentum.to_numpy(dtype=float),
                                        **{name:noise[:,i].astype(float) for i,name in enumerate(NOISE_NAMES)}}))
        return FeaturePacket(pd.concat(frames,ignore_index=True),{'returns':['momentum']},
                             {name:[name] for name in NOISE_NAMES},{name:0 for name in NOISE_NAMES})

    def predict_features(self, packet):
        frame = packet.frame
        return frame[['timestamp','market']].assign(target_position=np.sign(frame.momentum.to_numpy(dtype=float)))

    def predict(self, history, decision_times):
        return self.predict_features(self.feature_packet(history,decision_times))
