import numpy as np

from grader.score import block_evidence
from grader.uncertainty import score_standard_error, stationary_indices
from test_grader_score import execution_fixture


def test_stationary_bootstrap_has_fixed_seed_and_keeps_consecutive_day_runs():
    first=stationary_indices(200,7.,np.random.default_rng(17))
    second=stationary_indices(200,7.,np.random.default_rng(17))
    np.testing.assert_array_equal(first,second)
    assert ((first>=0)&(first<200)).all()
    assert np.mean(first[1:]==(first[:-1]+1)%200)>.7


def test_score_bootstrap_runs_all_thousand_replicates_with_intraday_cashflows(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,targets=execution_fixture(tmp_path,block,move_bps=30.,count=80)
        blocks.append(block_evidence(data,targets,block=block))
    uncertainty=score_standard_error([blocks,blocks],trial_sharpes=[-.2,.1],trial_count=6)
    assert uncertainty['resamples']==1000 and uncertainty['expected_block_days']==7.
    assert 0.<uncertainty['standard_error']<.5
    assert uncertainty['minimum']<=uncertainty['mean']<=uncertainty['maximum']
