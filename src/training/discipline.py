"""Training-only purged holdouts, stopping, and reproducible resampling."""
from copy import deepcopy

import numpy as np
import torch

from src.validation.splits import interval_coordinates, purge_training_spans


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0.):
        if patience < 1 or min_delta < 0:
            raise ValueError('patience must be positive and min_delta nonnegative')
        self.patience, self.min_delta = int(patience), float(min_delta)
        self.best_loss, self.best_epoch, self.bad_epochs = float('inf'), None, 0
        self.best_state = None

    def update(self, loss, epoch, model):
        if not np.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        if loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.bad_epochs = float(loss), int(epoch), 0
            self.best_state = deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def training_holdout(samples, indices, config, settings, *, dataset=None):
    """Reserve complete timestamp groups, then purge fit spans against the tail."""
    indices = np.asarray(indices, dtype=int)
    fraction = float(settings.get('validation_fraction', .2))
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    starts, _ = interval_coordinates(samples)
    ordered = indices[np.argsort(starts[indices], kind='stable')]
    groups = np.unique(starts[ordered])
    minimum = int(settings.get('min_rows', 2))
    count = max(1, int(np.ceil(len(groups) * fraction)))
    validation = ordered[starts[ordered] >= groups[-count]] if len(groups) else ordered
    proposed = np.setdiff1d(ordered, validation)
    audit = {'status': 'unavailable', 'reason': 'insufficient purged training/holdout rows',
             'scope': 'inside_training_partition_only', 'fit_indices': proposed.tolist(),
             'evaluation_indices': validation.tolist()}
    if minimum < 1:
        raise ValueError('min_rows must be positive')

    def unavailable():
        audit['attempted_fit_indices'] = audit['fit_indices']
        audit['attempted_evaluation_indices'] = audit['evaluation_indices']
        audit['fit_indices'] = indices.tolist()
        audit['evaluation_indices'] = []
        return indices, np.array([], dtype=int), audit

    if len(proposed) < minimum or len(validation) < minimum:
        return unavailable()
    data = dataset or {}
    clock = data.get('price_path', {}).get('timestamp')
    fit, purging = purge_training_spans(samples, proposed, validation, config.get('validation', {}),
                                       clock=clock, max_horizon=data.get('max_horizon_bars', 1),
                                       sequence_length=data.get('sequence_length', 1))
    audit.update(purging)
    if len(fit) < minimum:
        return unavailable()
    keys = [name for name in ['sample_index', 'horizon'] if name in samples]
    audit['fit_sample_keys'] = samples.iloc[fit][keys].to_dict('records')
    audit['evaluation_sample_keys'] = samples.iloc[validation][keys].to_dict('records')
    if data.get('coverage_config'):
        from src.data.side_data import validate_split_coverage
        lookup = {value: i for i, value in enumerate(data['samples']['sample_index'])}
        audit['data_coverage'] = validate_split_coverage(data, {'folds': [{'fold': 0,
            'train': np.array([lookup[value] for value in samples.iloc[fit]['sample_index'].unique()]),
            'validation': np.array([lookup[value] for value in samples.iloc[validation]['sample_index'].unique()])}]}, include_final=False)
    audit.update(status='completed', reason=None)
    return fit, validation, audit


def bootstrap_rows(samples, mode, seed):
    """Sequential bootstrap uses marginal uniqueness on inclusive span endpoints.

    Endpoint compression avoids an events-by-source-bars matrix. Draws are with
    replacement; only the supplied fit population participates.
    """
    n = len(samples)
    rng = np.random.default_rng(seed)
    if mode == 'none':
        return np.arange(n)
    if mode == 'bootstrap':
        return rng.choice(n, n, replace=True)
    if mode != 'sequential_bootstrap':
        raise ValueError(f'Unknown bootstrap mode: {mode}')
    if {'label_span_start_idx', 'label_span_end_idx'} <= set(samples):
        starts = samples['label_span_start_idx'].to_numpy(dtype=int)
        ends = samples['label_span_end_idx'].to_numpy(dtype=int)
        if (ends < starts).any():
            raise ValueError('Invalid bootstrap label intervals')
    else:
        starts, ends = interval_coordinates(samples)
    boundaries = np.unique(np.concatenate([starts, ends + 1]))
    left, right = np.searchsorted(boundaries, starts), np.searchsorted(boundaries, ends + 1)
    widths = np.diff(boundaries).astype(float)
    concurrency = np.zeros(len(widths))
    draws = []
    for _ in range(n):
        marginal = np.array([np.average(1. / (concurrency[a:b] + 1.), weights=widths[a:b])
                             for a, b in zip(left, right)])
        choice = int(rng.choice(n, p=marginal / marginal.sum()))
        draws.append(choice)
        concurrency[left[choice]:right[choice]] += 1
    return np.asarray(draws)


class SeedEnsemble(torch.nn.Module):
    def __init__(self, members, seeds):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.seeds = list(seeds)


class TabularBase(torch.nn.Module):
    """Sklearn heads on latest-bar and rolling mean/std summaries."""
    def __init__(self, regressors, classifiers):
        super().__init__()
        self.regressors, self.classifiers = regressors, classifiers

    @staticmethod
    def summary(x):
        return np.concatenate([x[:, -1], x.mean(axis=1), x.std(axis=1)], axis=1)

    def forward(self, x):
        matrix = self.summary(x.detach().cpu().numpy())
        returns = np.column_stack([head.predict(matrix) for head in self.regressors])
        probs = np.column_stack([np.full(len(matrix), head) if isinstance(head, float)
                                 else head.predict_proba(matrix)[:, 1] for head in self.classifiers])
        p = torch.as_tensor(np.clip(probs, 1e-7, 1 - 1e-7), dtype=x.dtype, device=x.device)
        return torch.as_tensor(returns, dtype=x.dtype, device=x.device), torch.logit(p)

    def get_extra_state(self):
        return {'regressors': self.regressors, 'classifiers': self.classifiers}

    def set_extra_state(self, state):
        self.regressors, self.classifiers = state['regressors'], state['classifiers']
