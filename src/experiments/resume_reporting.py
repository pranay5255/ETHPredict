"""Recover reporting from saved outcomes, without fitting or reading test again."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.evaluation.evidence import assess_evidence
from src.experiments.dashboard import build_dashboard
from src.experiments.meta_labeling_mvp import (
    _write_json, _report_markdown, _log_meta_label_trial_trackio, _selection_status_for_trial,
    _log_meta_label_summary_trackio, _log_accounting_summary_trackio, _stamp_trackio_receipts, _carry_selection_annotations, _final_test_ledger_path,
)
from src.utils.trackio_logging import _verify_local_run, trackio_enabled


def _immutable_outputs(report):
    paths = []
    for trial in report['trials']:
        paths += [Path(path) for name, path in trial.get('artifact_paths', {}).items()
                  if name in {'test_predictions', 'test_candidates', 'test_trades', 'oof_predictions', 'oof_candidates'}]
        if trial.get('model_path'):
            paths.append(Path(trial['model_path']))
    # Ledger bytes are also protected; reporting must not count a new evaluation.
    config = yaml.safe_load((Path(report['run_dir'])/'resolved_config.yml').read_text())
    ledger = _final_test_ledger_path(config, Path(report['run_dir']))
    if ledger.is_file():
        paths.append(ledger)
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def resume_reporting(manifest_path):
    manifest_path = Path(manifest_path)
    report = json.loads(manifest_path.read_text())
    run_dir = Path(report['run_dir'])
    receipt_path = run_dir / 'reporting_recovery.json'
    if receipt_path.exists():
        saved_receipt = json.loads(receipt_path.read_text())
        if saved_receipt.get('status') == 'completed':
            if _immutable_outputs(report) != saved_receipt['immutable_output_hashes']:
                raise ValueError('Protected outcomes changed after reporting recovery')
            return report
    before = _immutable_outputs(report)
    original = run_dir / 'report' / 'original_training_manifest.json'
    if not original.exists():
        original.write_text(manifest_path.read_text())
    config = yaml.safe_load((run_dir/'resolved_config.yml').read_text())
    assessment = assess_evidence(report, config)
    report.setdefault('legacy_presence_grade', report.get('evidence_grade'))
    report['evidence_grade'], report['evidence_assessment'] = assessment['grade'], assessment
    report['reporting_recovery'] = {'original_training_manifest': str(original),
        'original_manifest_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'predictions_or_fits_reexecuted': False, 'test_evaluations_added': 0,
        'reporting_source_sha256': {str(file.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(file.read_bytes()).hexdigest() for file in
            [Path(__file__), Path(__file__).with_name('dashboard.py'), Path(__file__).with_name('dashboard_template.html'), Path(__file__).parents[1]/'evaluation'/'evidence.py', Path(__file__).parents[1]/'utils'/'trackio_logging.py']},
        'evidence_note': 'Corrected report interpretation; original outcomes and ledger are byte-identical'}
    project = config.get('tracking', {}).get('trackio', {}).get('project', 'ethpredict')
    best, qualified, selection = report['best_trial'], report.get('best_trading_trial'), report['selection']
    for trial in report['trials']:
        name = f"meta_label/{report['run_id']}/{trial['trial_id']}"
        path = Path(trial['manifest_path']).parent/'trackio_receipt.json'
        reconciled = False
        if path.is_file() and trackio_enabled(config):
            old = json.loads(path.read_text())
            try:
                _verify_local_run(project, name, trial['status'])
            except Exception:
                pass
            else:
                _write_json(path, {**old, 'delivery': 'logged', 'verification': 'local_readback', 'reconciled_from_saved_run': True})
                reconciled = True
        if not reconciled:
            _log_meta_label_trial_trackio(config, run_id=report['run_id'], trial=trial,
                selection_status=_selection_status_for_trial(trial, best, qualified, selection),
                raw_best=best, best_trading=qualified, selection=selection, trial_count=len(report['trials']), smoke=report['smoke'])
    report['dashboard_path'] = str(build_dashboard(report))
    _log_meta_label_summary_trackio(config, run_id=report['run_id'], report=report, smoke=report['smoke'])
    summary = _log_accounting_summary_trackio(config, run_id=report['run_id'], run_dir=run_dir,
        accounting=report['trial_accounting'], smoke=report['smoke'], status=report['status'])
    _stamp_trackio_receipts(report['trials'], report['trial_accounting'], accounting_path=Path(report['trial_accounting_path']), summary_receipt=summary)
    report['best_trial'] = _carry_selection_annotations(best, report['trials'])
    _write_json(Path(report['best_trial_manifest_path']), report['best_trial'])
    if qualified is not None:
        report['best_trading_trial'] = _carry_selection_annotations(qualified, report['trials'])
        _write_json(Path(report['best_trading_trial_manifest_path']), report['best_trading_trial'])
    _write_json(Path(report['report_json_path']), report)
    Path(report['report_markdown_path']).write_text(_report_markdown(report))
    _write_json(manifest_path, report)
    after = _immutable_outputs(report)
    if before != after:
        raise AssertionError('Reporting recovery changed protected model/prediction/test artifacts')
    _write_json(receipt_path, {'status': 'completed', 'immutable_output_hashes': after,
                              'test_evaluations_added': 0, 'predictions_or_fits_reexecuted': False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    report = resume_reporting(args.manifest)
    print(json.dumps({'manifest': str(args.manifest), 'dashboard': report['dashboard_path'],
                      'grade': report['evidence_grade'], 'test_evaluations_added': 0}, indent=2))


if __name__ == '__main__':
    main()
