"""Bank/header validation with synthetic trusted-report stubs, not market runs."""
import json

import pytest

from grader import baseline_bank as bank
from grader.baselines import NAMES
from grader.data import DataContractError
from grader.models import sha256_file


def fixture_bank(tmp_path,monkeypatch):
    monkeypatch.setattr(bank,'grading_fingerprint',lambda:'fixture-contract')
    monkeypatch.setattr(bank,'read_report',lambda path:json.loads(path.read_text()))
    monkeypatch.setattr(bank,'verify_artifacts',lambda *args:None)
    records=[]
    for index,name in enumerate(NAMES):
        directory=tmp_path/name;directory.mkdir()
        metrics=[]
        for seed in [17,29]:
            blocks=[{'block':i,'net_1x_bps':float(index),'net_2x_bps':index/2.,'turnover':1.,'trades':1} for i in range(1,6)]
            metrics.append({'seed':seed,'trial_count':6,'blocks':blocks,'net_1x_bps':float(5*index),
                'net_2x_bps':2.5*index,'turnover':5.,'trades':5,
                'daily_sharpe_per_period':{'available':index>0,'value':index*.01 if index else None}})
        report={'mode':'baseline_calibration','tier':'dev','data':{'sha256':'fixture-data'},
            'strategy':{'declaration':{'resolution':'1h','markets':['ETH']}},'artifacts':{'manifest':{}},
            'metrics':metrics,'integrity':[{'name':name,'passed':name!='grader_hash'} for name in
                ['grader_hash','schema','causality','determinism','budget','tier_isolation','artifact_integrity']]}
        path=directory/'grade_report.json';path.write_text(json.dumps(report))
        records.append({'name':name,'report':path.relative_to(tmp_path).as_posix(),'sha256':sha256_file(path)})
    value={'version':1,'complete':True,'data_sha256':'fixture-data','grading_fingerprint':'fixture-contract',
        'markets':['ETH'],'resolution':'1h','tier':'dev','trial_count':6,
        'trial_sharpes':[i*.01 for i in range(1,6)],'winner':'lstm_recipe',
        'best_net_by_seed':{'17':25.,'29':25.},'records':records}
    path=tmp_path/'baseline_bank.json';path.write_text(json.dumps(value))
    return path,value


def test_bank_rederives_best_baseline_and_trial_sharpes_from_all_reports(tmp_path,monkeypatch):
    path,value=fixture_bank(tmp_path,monkeypatch)
    assert bank.read_bank(path)==value


@pytest.mark.parametrize('corruption',['winner','threshold','sharpes','blocks','integrity','markets','seed','report_hash'])
def test_changed_thresholds_and_incomplete_comparisons_fail_closed(tmp_path,monkeypatch,corruption):
    path,value=fixture_bank(tmp_path,monkeypatch)
    if corruption=='winner': value['winner']='flat'
    if corruption=='threshold': value['best_net_by_seed']['17']=-1e9
    if corruption=='sharpes': value['trial_sharpes'].append(1.)
    if corruption in {'blocks','integrity','markets','seed','report_hash'}:
        record=value['records'][0];report_path=tmp_path/record['report']
        report=json.loads(report_path.read_text())
        if corruption=='blocks': report['metrics'][0]['blocks']=[]
        if corruption=='integrity': report['integrity']=[]
        if corruption=='markets': report['strategy']['declaration']['markets']=['BTC']
        if corruption=='seed': report['metrics'][1]['seed']=17
        if corruption=='report_hash': report['metrics'][0]['net_1x_bps']=123.
        report_path.write_text(json.dumps(report))
        if corruption!='report_hash': record['sha256']=sha256_file(report_path)
    path.write_text(json.dumps(value))
    with pytest.raises(DataContractError): bank.read_bank(path)
