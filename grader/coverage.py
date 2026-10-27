"""Exact source coverage exception, without side-feature imputation helpers."""
import json

class DataCoverageError(ValueError):
    def __init__(self, message, report):
        self.report = report
        super().__init__(f'{message}: {json.dumps(report, default=str, sort_keys=True)}')
