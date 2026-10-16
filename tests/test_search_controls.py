from copy import deepcopy

import numpy as np
import pytest

from src.config.loader import expand_grid_search
from src.config.search import build_search_plan
from src.experiments.meta_labeling_mvp import select_trials_with_trade_floor
from src.evaluation.selection import comparison_budget


def test_grid_plan_records_full_population_and_executed_subset():
    config = {'search': {'mode': 'grid', 'max_trials': 2, 'spaces': {'base': {'x': [1, 2], 'y': [3, 4]}}}}
    plan = build_search_plan(config)
    assert plan['proposed_count'] == 4 and len(plan['expanded_space']) == 4
    assert [row['overrides'] for row in plan['executed_subset']] == [{'x': 1, 'y': 3}, {'x': 1, 'y': 4}]
    del config['search']['max_trials']
    assert len(expand_grid_search(config)) == 4


def test_shuffled_grid_is_seeded_and_does_not_take_grid_prefix():
    config = {'experiment': {'seed': 7}, 'search': {'mode': 'shuffled_grid', 'max_trials': 3,
              'spaces': {'base': {'x': list(range(100))}}}}
    first = build_search_plan(config)
    assert first == build_search_plan(config)
    assert [row['proposal_index'] for row in first['executed_subset']] != [0, 1, 2]
    assert len({row['proposal_index'] for row in first['executed_subset']}) == 3
    config['search']['seed'] = 8
    assert first['executed_subset'] != build_search_plan(config)['executed_subset']


@pytest.mark.parametrize('mode', ['random', 'log_uniform'])
def test_random_distributions_respect_frozen_bounds(mode):
    config = {'search': {'mode': mode, 'seed': 17, 'max_trials': 20, 'spaces': {'base': {
        'training.learning_rate': {'distribution': 'log_uniform', 'low': .00001, 'high': .01},
        'model.base.hidden_size': {'distribution': 'integer', 'low': 4, 'high': 16},
        'model.base.dropout': {'distribution': 'uniform', 'low': 0., 'high': .3},
        'model.base.type': ['linear', 'lstm']}}}}
    specs = expand_grid_search(config)
    assert specs == expand_grid_search(config)
    assert len(specs) == 20
    for spec in specs:
        assert .00001 <= spec['config']['training']['learning_rate'] <= .01
        assert 4 <= spec['config']['model']['base']['hidden_size'] <= 16
        assert 0 <= spec['config']['model']['base']['dropout'] <= .3
    invalid = deepcopy(config)
    invalid['search']['spaces']['base']['training.learning_rate']['low'] = 0.
    with pytest.raises(ValueError, match='positive'):
        expand_grid_search(invalid)


def test_meta_classification_and_minimizing_scores_ignore_missing_and_test_results():
    trials = [{'trial_id': name, 'metrics': {'validation': {'trades': trades, 'net_pnl': pnl,
               'meta_classification': {'f1': f1, 'calibration_error': ece}, 'selection': {'calibration_error': ece}},
               'test': {'net_pnl': test}}, 'folds': []}
              for name, trades, pnl, f1, ece, test in [('abstain', 0, 0., .9, None, 9999), ('pnl', 5, 5., .2, .3, 50), ('class', 6, 1., .8, .1, -100)]]
    config = {'pipeline': {'selection_metric': 'calibration_error', 'selection_mode': 'min', 'min_validation_trades': 2}}
    best, qualified, roles = select_trials_with_trade_floor(trials, config)
    assert best['trial_id'] == qualified['trial_id'] == 'class'
    assert roles['classification_best_trial_id'] == 'abstain'
    assert roles['calibration_best_trial_id'] == 'class'
    changed = deepcopy(trials)
    for row in changed:
        row['metrics']['test']['net_pnl'] *= -100
    assert select_trials_with_trade_floor(changed, config)[2] == roles
    assert comparison_budget(trials)['top_level_attempts'] == 3
