import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


class DataValidationError(Exception):
    """Raised when a CSV file does not conform to the schema."""


def validate_csvs(raw_dir: Path, schema_path: Path, rejects_dir: Path) -> List[Path]:
    """Validate CSVs against schema and return list of valid files."""
    schema = json.load(open(schema_path))
    required_cols = list(schema.keys())
    dtype_map = {k: np.dtype(v) for k, v in schema.items()}

    valid_files: List[Path] = []
    rejects_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(raw_dir.glob("*.csv")):
        try:
            chunks = pd.read_csv(
                csv_file,
                header=None,
                names=required_cols,
                usecols=range(len(required_cols)),
                chunksize=10000,
            )
            for chunk in chunks:
                for col in required_cols:
                    if chunk[col].dtype != dtype_map[col]:
                        raise DataValidationError(
                            f"Column {col} has {chunk[col].dtype}, expected {dtype_map[col]}"
                        )
            valid_files.append(csv_file)
        except Exception as exc:
            logging.error("CSV validation failed for %s: %s", csv_file, exc)
            (rejects_dir / csv_file.name).write_text("invalid")

    return valid_files
