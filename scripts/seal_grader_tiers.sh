#!/usr/bin/env bash
# No sudo here. The human invokes --apply as root after reviewing --plan.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${ETHPREDICT_SEAL_PYTHON:-python3}"
exec "$python_bin" "$script_dir/seal_grader_tiers.py" "$@"
