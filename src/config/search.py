"""Reconstructable seeded search plans with an explicit attempted-trial budget."""
from copy import deepcopy
from itertools import product
from math import prod

import numpy as np


MODES = {'grid', 'shuffled_grid', 'random', 'log_uniform'}


def _domains(search):
    output = {}
    for group in search.get('spaces', {}).values():
        if not isinstance(group, dict):
            raise ValueError('Each search space must map dotted parameter names to domains')
        for name, domain in group.items():
            if name in output:
                raise ValueError(f'Duplicate search parameter: {name}')
            output[name] = deepcopy(domain)
    return output


def _draw(domain, rng, mode):
    if not isinstance(domain, dict):
        values = domain if isinstance(domain, list) else [domain]
        if not values:
            raise ValueError('Search choice domains cannot be empty')
        return deepcopy(values[int(rng.integers(len(values)))])
    distribution = domain.get('distribution', 'log_uniform' if mode == 'log_uniform' else 'uniform')
    low, high = float(domain['low']), float(domain['high'])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError('Search bounds must be finite and ordered')
    if distribution == 'log_uniform':
        if low <= 0:
            raise ValueError('Log-uniform bounds must be positive')
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    if distribution == 'integer':
        if not low.is_integer() or not high.is_integer():
            raise ValueError('Integer bounds must be integers')
        return int(rng.integers(int(low), int(high) + 1))
    if distribution == 'uniform':
        return float(rng.uniform(low, high))
    raise ValueError(f'Unsupported search distribution: {distribution}')


def build_search_plan(config):
    search = config.get('search', {}) or {}
    mode = search.get('mode', 'grid')
    if mode not in MODES:
        raise ValueError(f'Unsupported search mode: {mode}')
    seed = int(search.get('seed', config.get('experiment', {}).get('seed', 42)))
    budget = int(search.get('max_trials', config.get('experiment', {}).get('trials', 1)))
    if budget < 1:
        raise ValueError('Search max_trials must be positive')
    rng = np.random.default_rng(seed)
    domains = _domains(search)
    explicit = search.get('trials')
    full = None
    if explicit:
        if mode not in {'grid', 'shuffled_grid'}:
            raise ValueError('Explicit trials support grid or shuffled_grid only')
        full = [{'id': str(row.get('id', f'grid_{i:03d}')), 'overrides': dict(row.get('overrides', {}))}
                for i, row in enumerate(explicit)]
        if len({row['id'] for row in full}) != len(full):
            raise ValueError('Explicit trial IDs must be unique')
        count = len(full)
        budget = budget if "max_trials" in search else count
        order = rng.permutation(count).tolist() if mode == 'shuffled_grid' else list(range(count))
        executed = [{**full[i], 'proposal_index': i} for i in order[:budget]]
    elif mode in {'grid', 'shuffled_grid'}:
        values = []
        for domain in domains.values():
            if isinstance(domain, dict):
                raise ValueError('Grid domains require discrete choices; use random/log_uniform for distributions')
            values.append(domain if isinstance(domain, list) else [domain])
        if any(not value for value in values):
            raise ValueError('Search choice domains cannot be empty')
        count = prod(map(len, values))
        budget = budget if "max_trials" in search else count
        if count > np.iinfo(np.int64).max:
            raise ValueError('Cartesian search space exceeds supported index range')
        order = (rng.choice(count, size=min(budget, count), replace=False).tolist()
                 if mode == 'shuffled_grid' else list(range(min(budget, count))))
        def proposal(index):
            positions = np.unravel_index(index, tuple(map(len, values))) if values else []
            return {key: deepcopy(domain[position]) for key, domain, position in zip(domains, values, positions)}
        executed = [{'id': f'grid_{i:03d}', 'proposal_index': original, 'overrides': proposal(original)} for i, original in enumerate(order)]
        if count <= int(search.get('expanded_space_limit', 10000)):
            full = [{'proposal_index': i, 'overrides': proposal(i)} for i in range(count)]
    else:
        count = None  # Continuous domains have no finite Cartesian population.
        executed = [{'id': f'{mode}_{i:03d}', 'proposal_index': i,
                     'overrides': {name: _draw(domain, rng, mode) for name, domain in domains.items()}}
                    for i in range(budget)]
    return {'schema_version': 1, 'mode': mode, 'seed': seed, 'max_trials': budget,
            'declared_domains': domains, 'parameter_order': list(domains), 'proposed_count': count,
            'expanded_space': full, 'reconstruction': 'declared domains in parameter_order; last parameter changes fastest',
            'executed_subset': executed, 'attempt_count': len(executed),
            'budget_semantics': 'attempts include failures, skips, repeats and no-trade outcomes; no replacement trials'}
