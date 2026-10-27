"""Purged whole-policy family permutations through numeric worker inputs.

The original vendored permutation_meta and summarize_importance calculate all
metrics/noise comparisons. The adapter supplies fresh subprocess predictions,
keeps permutations within each market, and never sends labels to the worker.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .data import DataContractError
from .feature_importance import permutation_meta, summarize_importance


def policy_fold_importance(runner, state, packet, declaration, reference, replay, block, *, repeats=3):
    if packet is None:
        return {'fold':block.block+1,'available':False,'reason':'Strategy does not expose feature_packet/predict_features',
                'permutation_rows':[],'auxiliary_comparisons':0,'split_audit':block.audit}
    frame = packet.frame
    times = pd.DatetimeIndex(frame.timestamp.drop_duplicates().sort_values())
    time_index = times.get_indexer(frame.timestamp)
    market_index = np.array([replay.markets.index(market) for market in frame.market])
    executed = time_index + replay.delay < replay.steps
    row_ids = np.flatnonzero(executed)
    if not len(row_ids):
        return {'fold':block.block+1,'available':False,'reason':'No signal executes inside this block',
                'permutation_rows':[],'auxiliary_comparisons':0,'split_audit':block.audit}
    truth = (replay.returns_bps[time_index[row_ids]+replay.delay,market_index[row_ids]] > 0).astype(float)
    selectors = pd.DataFrame({name:row_ids.copy() for name in packet.groups})
    selectors['meta_label'] = truth
    baseline_positions = reference.target_position.to_numpy()

    class PolicyProxy:
        kind = 'whole_strategy_policy'
        feature_columns = list(packet.groups)

        def predict_proba(self, selected):
            changed = frame.copy()
            identity = True
            for name, columns in packet.groups.items():
                keys = selected[name].to_numpy()
                if not np.issubdtype(keys.dtype,np.integer) or len(np.unique(keys)) != len(row_ids):
                    raise DataContractError('Invalid permutation selector')
                identity &= np.array_equal(keys,row_ids)
                source = row_ids.copy()
                for market in declaration.markets:
                    within = frame.iloc[row_ids].market.to_numpy() == market
                    eligible = row_ids[within]
                    # The vendored global permutation provides random keys.
                    # Their ranks give a proper uniform permutation within each
                    # market, preserving its price/feature scale distribution.
                    ranks = np.argsort(np.argsort(keys[within],kind='stable'),kind='stable')
                    source[within] = eligible[ranks]
                changed.loc[row_ids,columns] = frame.iloc[source][columns].to_numpy()
            positions = runner.predict_features(state,replace(packet,frame=changed),declaration)
            values = positions.target_position.to_numpy()
            if identity and not np.allclose(values,baseline_positions,rtol=0,atol=1e-9):
                raise DataContractError('Isolated input-only prediction differs from the actual policy')
            return (values[row_ids]+1.)/2.

    report = permutation_meta(PolicyProxy(),selectors,repeats=repeats,
                              seed=runner.seed+block.block+1,threshold=.5)
    rows = [{**row,'fold':block.block+1} for row in report['rows']]
    return {'fold':block.block+1,'available':report['available'],'reason':report['reason'],
        'permutation_rows':rows,'auxiliary_comparisons':0,'split_audit':block.audit,
        'fit_indices':block.train.tolist(),'evaluation_indices':block.evaluation.tolist(),
        'scored_policy_rows':row_ids.tolist(),'nonexecuted_rows':np.flatnonzero(~executed).tolist(),
        'families':packet.families,'noise':packet.noise,'baseline':report.get('baseline'),
        'target':'next execution-interval return direction',
        'probability_proxy':'(actual target_position + 1) / 2, clipped only by vendored log-loss routine',
        'permutation_scope':'within-market whole-family input windows; no evaluation labels or raw history in worker'}


def summarize_policy_importance(folds):
    completed = [fold for fold in folds if fold['available']]
    report = summarize_importance(completed)
    if len(completed) != 5:
        report['status'] = 'unavailable'
        report['unavailable_folds'] = [fold for fold in folds if not fold['available']]
    report['method'] = 'vendored permutation_meta and summarize_importance with isolated whole-policy input adapter'
    return report
