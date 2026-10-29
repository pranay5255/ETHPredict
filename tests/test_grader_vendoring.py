import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grader import stats as frozen_stats
from grader.costs import CostModel as FrozenCosts
from src.evaluation import backtest_stats as source_stats
from src.simulator.costs import CostModel as SourceCosts


def test_vendored_sources_and_recorded_adaptations_match_actual_bytes():
    root=Path(__file__).parents[1]
    records=json.loads((root/'grader/vendor_sources.json').read_text())
    for target,record in records.items():
        assert hashlib.sha256((root/record['source']).read_bytes()).hexdigest()==record['sha256']
        expected=record.get('vendored_sha256',record['sha256'])
        assert hashlib.sha256((root/target).read_bytes()).hexdigest()==expected
    for path in (root/'grader').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            assert all(name.split('.')[0] not in {'src','strategy'} for name in names),path


def test_vendored_cost_and_statistics_numerical_parity(tmp_path):
    clock=pd.date_range('2025-09-01',periods=24,freq='h',tz='UTC')
    funding=tmp_path/'funding.csv'
    rng=np.random.default_rng(4)
    pd.DataFrame({'timestamp':clock,'rate':rng.normal(0,.1,24)}).to_csv(funding,index=False)
    path=pd.DataFrame({'timestamp':clock,'high':102.,'low':99.,'close':100.})
    config={'mode':'dynamic','fee_bps':4.,'spread_bps':2.,'slippage_bps':1.,
            'dynamic_spread':{'enabled':True},
            'dynamic_funding':{'enabled':True,'historical':True,'paths':[str(funding)],'rate_column':'rate','rate_unit':'bps'},
            'execution':{'mode':'hybrid','fill_policy':'sampled','entry_delay_bars':2}}
    source,frozen=SourceCosts(config,path,granularity='1h'),FrozenCosts(config,path,granularity='1h')
    for i in range(12):
        signal={'sample_index':i,'entry_idx':i,'horizon_bars':4,'side':1 if i%2 else -1}
        assert source.quote(signal)==frozen.quote(signal)
        assert source.quote(signal,exit_index=i+4)==frozen.quote(signal,exit_index=i+4)
        assert source.fill(signal)==frozen.fill(signal)
    returns=rng.normal(.001,.01,200)
    for name,args in [('sharpe_ratio',(returns,)),('probabilistic_sharpe_ratio',(returns,)),
                      ('deflated_sharpe_ratio',(returns,[.1,.2,-.1],8)),
                      ('underwater_statistics',(np.cumsum(returns),)),('hhi_concentration',(returns,))]:
        assert getattr(source_stats,name)(*args)==getattr(frozen_stats,name)(*args)
