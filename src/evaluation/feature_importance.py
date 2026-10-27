"""Purged outer-validation diagnostics, never final-test feature discovery."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.data.features_all import FEATURE_FAMILY_COLUMNS


def binary_scores(truth, probability, threshold=.5):
    truth, probability = np.asarray(truth, dtype=float), np.asarray(probability, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(probability)
    truth, probability = truth[valid], np.clip(probability[valid], 1e-7, 1-1e-7)
    if not len(truth):
        return {'f1': None, 'negative_log_loss': None}
    return {'f1': float(f1_score(truth, probability >= threshold, zero_division=0)),
            'negative_log_loss': -float(log_loss(truth, probability, labels=[0, 1]))}


def permutation_meta(model, frame, *, repeats=3, seed=42, threshold=.5):
    if 'prediction_role' in frame and frame.prediction_role.eq('final_test').any():
        raise ValueError('Feature importance cannot read final-test outcomes')
    if model.kind == 'constant' or frame.empty:
        return {'available': False, 'reason': 'constant meta model or empty held-out labels', 'rows': []}
    baseline = binary_scores(frame.meta_label, model.predict_proba(frame), threshold)
    rng = np.random.default_rng(seed)
    rows = []
    for feature in model.feature_columns:
        values = {name: [] for name in baseline}
        for _ in range(repeats):
            changed = frame.copy()
            changed[feature] = rng.permutation(changed[feature].to_numpy())
            score = binary_scores(frame.meta_label, model.predict_proba(changed), threshold)
            for metric in baseline:
                values[metric].append(baseline[metric] - score[metric])
        rows.append({'feature': feature, 'model': 'meta', 'method': 'permutation',
                     'is_noise': feature.startswith('diagnostic_noise_'),
                     **{metric + '_drop': float(np.mean(value)) for metric, value in values.items()},
                     **{metric + '_std': float(np.std(value)) for metric, value in values.items()}})
    return {'available': True, 'reason': None, 'baseline': baseline, 'rows': rows}


def base_scores(predictions, horizons):
    scores = [binary_scores(predictions[f'{name}_true_direction'], predictions[f'{name}_direction_prob']) for name in horizons]
    return {metric: float(np.mean([row[metric] for row in scores])) for metric in ['f1', 'negative_log_loss']}


def subset_features(dataset, columns):
    positions = [dataset['feature_columns'].index(name) for name in columns]
    subset = dict(dataset)
    subset.update(X=dataset['X'][:, :, positions], input_size=len(positions), feature_columns=list(columns))
    subset['feature_manifest'] = {**dataset.get('feature_manifest', {}), 'columns': list(columns), 'diagnostic_subset': True}
    return subset


def fold_importance(base_model, meta_model, dataset, fold, candidates, config, device):
    from src.experiments.meta_labeling_mvp import predict_base_model, train_base_model
    settings = config.get('feature_importance', {}) or {}
    repeats, seed = int(settings.get('repeats', 3)), int(settings.get('seed', 42)) + int(fold['fold'])
    if repeats < 1:
        raise ValueError('Importance repeats must be positive')
    # These are the exact purged populations used by the parent outer fit.
    train, validation = np.asarray(fold['train']), np.asarray(fold['validation'])
    if set(train) & set(validation):
        raise ValueError('Feature importance training and validation overlap')
    baseline_predictions = predict_base_model(base_model, dataset, validation, device)
    baseline = base_scores(baseline_predictions, dataset['horizon_names'])
    columns = list(dataset['feature_columns'])
    selected = list(settings.get('base_features', columns[:int(settings.get('max_base_features', 6))]))
    selected += [name for name in columns if name.startswith('diagnostic_noise_') and name not in selected]
    if set(selected) - set(columns):
        raise ValueError('Importance base_features must be selected model columns')
    rng = np.random.default_rng(seed)
    rows = []
    for feature in selected:
        drops = {metric: [] for metric in baseline}
        for _ in range(repeats):
            modified = dict(dataset)
            modified['X'] = dataset['X'].clone()
            permutation = rng.permutation(validation)
            index = columns.index(feature)
            modified['X'][validation, :, index] = dataset['X'][permutation, :, index]
            score = base_scores(predict_base_model(base_model, modified, validation, device), dataset['horizon_names'])
            for metric in baseline:
                drops[metric].append(baseline[metric] - score[metric])
        rows.append({'feature': feature, 'model': 'base', 'method': 'permutation', 'fold': int(fold['fold']),
                     'is_noise': feature.startswith('diagnostic_noise_'),
                     **{metric + '_drop': float(np.mean(values)) for metric, values in drops.items()},
                     **{metric + '_std': float(np.std(values)) for metric, values in drops.items()}})
    labeled = candidates[candidates.is_candidate & candidates.meta_label.notna()].copy()
    if 'meta_evaluation_status' in labeled:
        labeled = labeled[labeled.meta_evaluation_status == 'held_out']
    threshold = getattr(meta_model, 'decision_threshold', None) or float(config.get('alpha_backtest', {}).get('meta_threshold', .5))
    meta = permutation_meta(meta_model, labeled, repeats=repeats, seed=seed, threshold=threshold)
    rows.extend([{**row, 'fold': int(fold['fold'])} for row in meta['rows']])
    comparisons = []
    specifications = []
    if settings.get('single_feature', False):
        specifications += [('single_feature', name, [name]) for name in selected]
    for family in settings.get('ablation_groups', []):
        if family not in config.get('features', {}).get('include', []):
            raise ValueError(f'Ablation group {family} must be predeclared in features.include')
        group = FEATURE_FAMILY_COLUMNS.get(family, [family])
        retained = [name for name in columns if name not in group]
        if not retained or len(retained) == len(columns):
            raise ValueError(f'Ablation {family} removes no features or all features')
        specifications.append(('group_ablation', family, retained))
    for method, name, retained in specifications:
        subset = subset_features(dataset, retained)
        trained, history = train_base_model(subset, train, config, device)
        scores = base_scores(predict_base_model(trained, subset, validation, device), dataset['horizon_names'])
        comparisons.append({'method': method, 'feature': name, 'retained_columns': retained, 'model': 'base',
                            'fold': int(fold['fold']), 'scores': scores, 'training_history': history,
                            'change_from_full': {metric: scores[metric] - baseline[metric] for metric in baseline}})
    mdi = []
    model = getattr(base_model, 'members', [base_model])[0]
    for head in getattr(model, 'regressors', []):
        if hasattr(head, 'feature_importances_'):
            reduced = np.asarray(head.feature_importances_).reshape(3, len(columns)).sum(axis=0)
            mdi.append(dict(zip(columns, reduced.tolist())))
    meta_manifest = meta_model.manifest()
    return {'fold': int(fold['fold']), 'scope': 'purged_outer_validation_only',
            'fit_indices': train.tolist(), 'evaluation_indices': validation.tolist(), 'split_audit': fold.get('audit'),
            'base': {'baseline': baseline, 'permutation_features': selected, 'mdi': mdi}, 'meta': meta,
            'permutation_rows': rows, 'ablations': comparisons,
            'meta_interpretability': meta_manifest.get('interpretability'), 'auxiliary_comparisons': len(comparisons)}


def summarize_importance(folds):
    rows = [row for fold in folds for row in fold['permutation_rows']]
    summary = []
    if rows:
        frame = pd.DataFrame(rows)
        for (model, name), part in frame.groupby(['model', 'feature'], sort=True):
            value, std = float(part.negative_log_loss_drop.mean()), float(part.negative_log_loss_drop.std(ddof=0))
            noise = bool(part.is_noise.any())
            category = 'noise_control' if noise else 'unstable' if std > abs(value) else 'positive' if value > 0 else 'negative_or_redundant'
            summary.append({'model': model, 'feature': name, 'negative_log_loss_drop': value, 'fold_std': std,
                            'f1_drop': float(part.f1_drop.mean()), 'folds': len(part), 'category': category})
    noise_failures = []
    for model in ['base', 'meta']:
        real = [row['negative_log_loss_drop'] for row in summary if row['model'] == model and row['category'] != 'noise_control']
        for row in summary:
            if row['model'] == model and row['category'] == 'noise_control' and row['negative_log_loss_drop'] > max([0., *real]):
                noise_failures.append(row['feature'])
    return {'status': 'completed' if folds else 'unavailable', 'scope': 'purged_outer_validation_only', 'folds': folds,
            'summary': summary, 'noise_control_failures': noise_failures,
            'auxiliary_comparisons': sum(fold['auxiliary_comparisons'] for fold in folds),
            'interpretation': 'validation diagnostics; positive importance alone does not establish a causal or deployable edge'}


def write_importance(report, directory, write_json):
    directory = Path(directory)
    paths = {'feature_importance': directory / 'feature_importance_validation.json',
             'feature_importance_table': directory / 'feature_importance_validation.parquet',
             'feature_importance_summary': directory / 'feature_importance_validation.md',
             'feature_ablation_table': directory / 'feature_ablations_validation.parquet'}
    write_json(paths['feature_importance'], report)
    pd.DataFrame(report['summary']).to_parquet(paths['feature_importance_table'], index=False)
    ablations = [{key: value for key, value in row.items() if key not in {'training_history', 'scores', 'change_from_full'}}
                 | row['scores'] | {'delta_' + key: value for key, value in row['change_from_full'].items()}
                 for fold in report['folds'] for row in fold['ablations']]
    pd.DataFrame(ablations).to_parquet(paths['feature_ablation_table'], index=False)
    lines = ['# Purged validation feature diagnostics', '', report['interpretation'], '',
             f"Noise controls ranked above all evaluated real features: {report['noise_control_failures']}", '']
    lines += [f"- {row['model']} / {row['feature']}: {row['category']}; log-loss usefulness {row['negative_log_loss_drop']:.6f}; fold std {row['fold_std']:.6f}" for row in report['summary']]
    paths['feature_importance_summary'].write_text('\n'.join(lines) + '\n')
    return paths
