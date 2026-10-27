"""Validation-only scores. Missing evidence is unavailable, never a perfect score."""
import numpy as np
import pandas as pd

ALIASES = {'validation_f1': 'metrics.validation.selection.f1',
           'negative_log_loss': 'metrics.validation.selection.negative_log_loss',
           'calibration_error': 'metrics.validation.selection.calibration_error',
           'edge_rank_correlation': 'metrics.validation.selection.edge_rank_correlation',
           'net_pnl': 'metrics.validation.net_pnl', 'composite': 'metrics.validation.selection.composite'}


def validation_selection_scores(candidates, metrics, config):
    frame = candidates[candidates.is_candidate & candidates.meta_label.notna()].copy()
    if 'meta_evaluation_status' in frame:
        frame = frame[frame.meta_evaluation_status == 'held_out']
    if 'prediction_role' in frame and frame.prediction_role.eq('final_test').any():
        raise ValueError('Selection scores cannot use final-test outcomes')
    classification = metrics.get('meta_classification') or {}
    f1, loss, ece = (classification.get(name) for name in ['f1', 'log_loss', 'calibration_error'])
    edge_rank = None
    if len(frame) > 1 and frame.expected_edge_bps.std() > 0 and frame.label_net_return.std() > 0:
        correlation = frame.expected_edge_bps.corr(frame.label_net_return, method='spearman')
        edge_rank = float(correlation) if np.isfinite(correlation) else None
    pipeline = config.get('pipeline', {})
    minimum = int(pipeline.get('min_validation_trades', 1))
    weights = pipeline.get('composite_weights', {'f1': 1., 'calibration': .5, 'edge_rank': .1})
    qualified = metrics.get('trades', 0) >= minimum
    composite = None
    if qualified and f1 is not None and ece is not None:
        composite = float(weights.get('f1', 1.) * f1 - weights.get('calibration', .5) * ece
                          + weights.get('edge_rank', .1) * (edge_rank if edge_rank is not None else 0.))
    return {'f1': f1, 'negative_log_loss': -loss if loss is not None else None,
            'calibration_error': ece, 'edge_rank_correlation': edge_rank, 'composite': composite,
            'trade_qualified': qualified, 'evaluated_candidate_rows': len(frame),
            'source': 'held_out_outer_validation_only',
            'composite_unavailable_reason': None if composite is not None else 'insufficient trades or classification evidence'}


def comparison_budget(trials):
    """Conservative count of disclosed model/policy comparisons, not fit calls."""
    nested, thresholds, ablations = 0, 0, 0
    for trial in trials:
        audits = [row.get('meta_validation', {}).get('model', {}) for row in trial.get('folds', [])]
        audits.append(trial.get('meta_labeler', {}))
        for model in audits:
            nested += max(0, len(model.get('model_selection', {}).get('scores', [])) - 1)
            if model.get('threshold_selection', {}).get('source') == 'purged_meta_training_holdout_only':
                thresholds += max(0, len(model.get('threshold_selection', {}).get('table', [])) - 1)
        ablations += int(trial.get('feature_importance', {}).get('auxiliary_comparisons', 0))
    return {'top_level_attempts': len(trials), 'nested_model_comparisons': nested,
            'nested_threshold_comparisons': thresholds, 'diagnostic_ablations': ablations,
            'conservative_comparison_count': len(trials) + nested + thresholds + ablations,
            'scope': 'disclosed within-run comparisons; dependent comparisons counted conservatively, not independent trials',
            'prior_research_attempts': 'not estimated; archived runs remain separate evidence'}
