import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from grader.baselines import materialize
from grader.checks import causality_check
from grader.models import stage_models
from grader.runner import StrategyRunner
from test_grader_evaluation_data import fixture_data


def test_registered_timesfm_median_baseline_runs_real_cuda_and_twenty_prefixes(tmp_path):
    model=os.environ.get('ETHPREDICT_TIMESFM_MODEL_PATH')
    if model is None or not torch.cuda.is_available():
        pytest.skip('Pinned TimesFM checkpoint and CUDA are required for this integration')
    registered=stage_models(['timesfm-2.5'],{'timesfm-2.5':model},tmp_path/'models')
    data=fixture_data(160).candles
    source=materialize('timesfm_zero_shot',tmp_path/'source',markets=['ETH'],resolution='1h')
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[128:],unit='ms',utc=True))
    with StrategyRunner(source,tmp_path/'run',seed=17,device='cuda',registered_model_paths=registered) as runner:
        declaration=runner.declare()
        state=runner.fit(data)
        assert state.stat().st_size<1_048_576  # immutable backbone is referenced, not duplicated
        positions=runner.predict(state,data,times,declaration)
        assert len(positions)==32 and positions.target_position.isin([-1.,0.,1.]).all()
        check=causality_check(runner,state,data,times,declaration,positions,block=1,seed=17)
        assert check.passed and len(check.details['cutoffs_ms'])==20
        assert max(call.peak_vram_bytes for call in runner.calls)>0
