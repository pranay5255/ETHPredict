"""Descriptive held-out forecast diagnostics and configurable signal policies."""
import numpy as np
import pandas as pd


MAPPING_MODES = {'raw_sign', 'probability_sign', 'expected_edge', 'volatility_scaled_edge', 'horizon_agreement'}


def calibration_summary(truth, probability, bins=10):
    y, p = np.asarray(truth, dtype=float), np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], np.clip(p[valid], 0., 1.)
    groups = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    for bucket in range(bins):
        mask = groups == bucket
        table.append({'lower': bucket / bins, 'upper': (bucket + 1) / bins, 'count': int(mask.sum()),
                      'mean_probability': float(p[mask].mean()) if mask.any() else None,
                      'observed_rate': float(y[mask].mean()) if mask.any() else None})
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {'log_loss': float(-np.mean(y * np.log(clipped) + (1-y) * np.log(1-clipped))) if len(y) else None,
            'count': len(y), 'brier_score': float(np.mean((p-y)**2)) if len(y) else None,
            'ece': float(sum(row['count'] / len(y) * abs(row['mean_probability'] - row['observed_rate'])
                             for row in table if row['count'])) if len(y) else None,
            'reliability': table}


def map_forecast(pred_return, direction_prob, volatility, disagreement, cost_bps, min_edge_bps, settings=None):
    settings = settings or {}
    mode = settings.get('mode', 'expected_edge')
    if mode not in MAPPING_MODES:
        raise ValueError(f'Unknown candidate mapping: {mode}')
    side = int(np.sign(direction_prob - .5) if mode == 'probability_sign' else np.sign(pred_return))
    edge = side * pred_return * 10000. - cost_bps
    scaled = edge / max(abs(volatility) * 10000., 1e-8)
    eligible = side != 0
    if mode in {'expected_edge', 'horizon_agreement'}:
        eligible &= edge >= min_edge_bps
    if mode == 'volatility_scaled_edge':
        eligible &= edge >= min_edge_bps and scaled >= float(settings.get('min_scaled_edge', 0.))
    if mode == 'horizon_agreement':
        eligible &= disagreement == 0.
    return {'side': side, 'expected_edge_bps': edge, 'volatility_scaled_edge': scaled,
            'is_candidate': bool(eligible), 'candidate_mapping': mode}


def forecast_diagnostics(frame, horizon, cost_bps=0.):
    true = frame[f'{horizon}_true_return'].to_numpy(dtype=float)
    pred = frame[f'{horizon}_pred_return'].to_numpy(dtype=float)
    prob = frame[f'{horizon}_direction_prob'].to_numpy(dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred) & np.isfinite(prob)
    true, pred, prob = true[valid], pred[valid], prob[valid]
    cost_bps = np.broadcast_to(np.asarray(cost_bps, dtype=float), len(frame))[valid]
    rank = None
    if len(true) > 1 and np.ptp(pred) > 0 and np.ptp(true) > 0:
        rank = float(pd.Series(pred).corr(pd.Series(true), method='spearman'))
    slope, intercept = (None, None)
    if len(true) > 1 and np.std(pred) > 1e-12:
        slope = float(np.mean((pred - pred.mean()) * (true - true.mean())) / np.var(pred))
        intercept = float(true.mean() - slope * pred.mean())
    side = np.sign(pred)
    gross = side * true * 10000.
    net = gross - (side != 0) * cost_bps
    edge = np.abs(pred) * 10000. - cost_bps
    table = pd.DataFrame({'edge': edge, 'gross': gross, 'net': net, 'active': side != 0})
    table['bucket'] = pd.cut(edge, [-np.inf, 0, 5, 10, 25, 50, np.inf], labels=['lt_0', '0_5', '5_10', '10_25', '25_50', '50_plus'])
    buckets = []
    for label in table.bucket.cat.categories:
        part = table[table.bucket == label]
        buckets.append({'bucket': str(label), 'count': len(part), 'coverage': len(part) / max(len(table), 1),
                        'mean_predicted_edge_bps': float(part.edge.mean()) if len(part) else None,
                        'mean_realized_gross_edge_bps': float(part.gross.mean()) if len(part) else None,
                        'mean_realized_net_edge_bps': float(part.net.mean()) if len(part) else None,
                        'nonzero_side_count': int(part.active.sum())})
    edge_rank = None
    if len(edge) > 1 and np.std(edge) > 1e-12 and np.std(net) > 1e-12:
        edge_rank = float(pd.Series(edge).corr(pd.Series(net), method='spearman'))
    return {'calibration': calibration_summary(true > 0, prob),
            'return_calibration': {'slope': slope, 'intercept': intercept, 'scope': 'descriptive_evaluation_only'},
            'rank_correlation': {'spearman': rank, 'edge_to_realized_net_spearman': edge_rank, 'reason': None if rank is not None else 'insufficient or constant values'},
            'edge_ranking': {'cost_bps': float(cost_bps.mean()) if len(cost_bps) else None,
                             'cost_bps_min': float(cost_bps.min()) if len(cost_bps) else None,
                             'cost_bps_max': float(cost_bps.max()) if len(cost_bps) else None, 'predicted_edge_buckets': buckets,
                             'mean_realized_gross_edge_bps': float(gross.mean()) if len(gross) else None,
                             'mean_realized_net_edge_bps': float(net.mean()) if len(net) else None,
                             'candidate_coverage': float(np.mean((side != 0) & (edge >= 0))) if len(edge) else 0.}}
