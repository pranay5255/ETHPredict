"""Trusted numeric decisions and a private-only overfit monitor."""
from __future__ import annotations

from copy import deepcopy
import math

from grader.grade import EvaluationResult, rescore
from grader.artifacts import verify_artifacts
from grader.report import load_evidence, read_report


def rescore_saved(report_path, history, bank):
    """Recompute DSR at today's count, including interrupted/crashed attempts."""
    report = deepcopy(read_report(report_path))
    verify_artifacts(report_path.parent, report['artifacts']['manifest'])
    blocks = [[load_evidence(path) for path in seed] for seed in report['artifacts']['seed_blocks']]
    result = EvaluationResult(report, blocks)
    rescore(result, trial_sharpes=[*bank['trial_sharpes'], *history['sharpes']],
        trial_count=bank['trial_count'] + history['count'],
        baseline_net_by_seed=bank['best_net_by_seed'])
    return report


def decision(candidate, incumbent=None):
    ready = (len(candidate['metrics']) == 2 and
        all(len(row['blocks']) == 5 for row in candidate['metrics']) and
        bool(candidate['integrity']) and all(row['passed'] for row in candidate['integrity']))
    if incumbent is None:
        # Establish a measured control, even if its economic score is zero.
        # This initialization is not an improvement claim.
        return {'decision': 'initialize' if ready else 'discard',
                'incumbent_rescored_score': None, 'candidate_score': candidate['score'], 'delta': None}
    uncertainty = candidate.get('uncertainty')
    delta = uncertainty.get('standard_error') if uncertainty is not None else None
    if delta is not None and (isinstance(delta, bool) or not isinstance(delta, (int, float))
                              or not math.isfinite(delta) or delta < 0):
        delta = None
    keep = (ready and candidate['status'] == 'accepted' and delta is not None
            and candidate['score'] > incumbent['score'] + delta)
    return {'decision': 'keep' if keep else 'discard',
            'incumbent_rescored_score': incumbent['score'],
            'candidate_score': candidate['score'], 'delta': delta}


def monitor(previous, *, trial_count, dev_score, hidden_score):
    """No value from this state may enter a response, public log or Trackio."""
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in [dev_score, hidden_score]):
        raise ValueError('Private monitor requires valid scores')
    state = deepcopy(previous) if previous is not None else {'observations': [], 'increases': 0, 'alert': False}
    if state['observations'] and trial_count <= state['observations'][-1]['trial_count']:
        return state
    # The initial measured incumbent supplies a reference. Checks at 10/20/30
    # can then establish three actual consecutive increases rather than just
    # three observations (only two changes).
    if state['observations'] and trial_count % 10:
        return state
    gap = dev_score - hidden_score
    if state['observations']:
        state['increases'] = state['increases'] + 1 if gap > state['observations'][-1]['gap'] else 0
    state['observations'].append({'trial_count': trial_count, 'gap': gap})
    state['alert'] = state['alert'] or state['increases'] >= 3
    return state
