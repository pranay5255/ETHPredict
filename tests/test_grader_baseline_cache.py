import json

import pytest

from grader import baseline_cache as cache
from grader.data import DataContractError
from grader.grade import data_identity
from test_grader_evaluation_data import declaration, fixture_data


def install_builder(monkeypatch):
    calls=[]
    def build(lake,directory,**kwargs):
        calls.append(kwargs)
        directory.mkdir()
        data=fixture_data()
        value={'complete':True,'markets':kwargs['markets'],'resolution':kwargs['resolution'],
               'tier':kwargs['tier'],'data_sha256':data_identity(data)['sha256']}
        (directory/'baseline_bank.json').write_text(json.dumps(value))
        return value
    monkeypatch.setattr(cache,'build_bank',build)
    monkeypatch.setattr(cache,'read_bank',lambda path:json.loads(path.read_text()))
    monkeypatch.setattr(cache,'grading_fingerprint',lambda:'frozen-contract')
    return calls


def test_missing_bank_is_built_once_and_reused_with_stable_artifact_paths(tmp_path,monkeypatch):
    calls=install_builder(monkeypatch)
    config,data=declaration(),fixture_data()
    first=cache.get_bank(None,tmp_path,config,data,device='cpu')
    assert cache.get_bank(None,tmp_path,config,data,device='cpu')==first and len(calls)==1
    index=next(tmp_path.glob('*.json'))
    record=json.loads(index.read_text())
    assert (tmp_path/record['bank']).is_file()
    assert not list(tmp_path.glob('*.tmp'))


def test_contract_change_uses_a_new_bank_and_preserves_previous_attempt(tmp_path,monkeypatch):
    calls=install_builder(monkeypatch)
    cache.get_bank(None,tmp_path,declaration(),fixture_data(),device='cpu')
    monkeypatch.setattr(cache,'grading_fingerprint',lambda:'new-contract')
    cache.get_bank(None,tmp_path,declaration(),fixture_data(),device='cpu')
    assert len(calls)==2 and len(list(tmp_path.glob('*/baseline_bank.json')))==2


@pytest.mark.parametrize('corruption',['bank','index','escape'])
def test_corrupt_or_escaping_cache_is_rejected_without_silent_rebuild(tmp_path,monkeypatch,corruption):
    calls=install_builder(monkeypatch)
    cache.get_bank(None,tmp_path,declaration(),fixture_data(),device='cpu')
    index=next(tmp_path.glob('*.json'))
    value=json.loads(index.read_text())
    if corruption=='bank': (tmp_path/value['bank']).write_text('{}')
    if corruption=='index': value['unknown']=True
    if corruption=='escape': value['bank']='../outside.json'
    index.write_text(json.dumps(value))
    with pytest.raises(DataContractError): cache.get_bank(None,tmp_path,declaration(),fixture_data(),device='cpu')
    assert len(calls)==1


def test_incomplete_build_keeps_evidence_and_publishes_no_success_index(tmp_path,monkeypatch):
    calls=install_builder(monkeypatch)
    original=cache.build_bank
    def incomplete(*args,**kwargs):
        value=original(*args,**kwargs);value['complete']=False
        return value
    monkeypatch.setattr(cache,'build_bank',incomplete)
    with pytest.raises(DataContractError,match='retained evidence'):
        cache.get_bank(None,tmp_path,declaration(),fixture_data(),device='cpu')
    assert len(calls)==1 and not list(tmp_path.glob('*.json'))
    assert len(list(tmp_path.glob('*/baseline_bank.json')))==1


def test_baseline_comparability_checks_all_markets_clock_tier_and_data():
    config,data=declaration(),fixture_data()
    bank={'data_sha256':data_identity(data)['sha256'],'tier':'dev','resolution':'1h','markets':['ETH']}
    assert cache.comparable_bank(bank,config,data)==bank
    for bad in [dict(bank,markets=['ETH','BTC']),dict(bank,resolution='5m'),dict(bank,tier='shadow'),dict(bank,data_sha256='changed')]:
        with pytest.raises(DataContractError): cache.comparable_bank(bad,config,data)
