# Project documentation and local-work verification — 2026-10-17

Review base: merged main `d09efad`. This series preserves the pre-existing local
EDA/notebook/dependency work and updates the active project documentation for the
completed lake. Each changed path is committed separately, including previously
staged files and the two previously staged document deletions. PR #51 and draft
PR #53 remain unmerged; their software/results are described with branch scope.

## Checks executed in the primary checkout

- `OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **268 passed, 39 warnings**
  in 102.23 seconds. Warnings concern SDK/websockets deprecations and numerical
  behavior in synthetic statistical fixtures. This is a software regression run,
  not a new performance experiment.
- `node tests/test_eda_frontend.cjs`: passed compatibility-key and split checks.
- `uv lock --check`: passed, resolving 231 packages.
- Both research notebooks: nbformat schema validation passed; every code cell's
  saved outputs and execution count are identical to the original staged version.
  Notebook cells were not executed by this documentation update.
- Python syntax checks: backfill collector, EDA script and notebook executor passed.
  `scripts/lighter_backfill_lake.py --help` passed without collection/API requests.
- Independent read-only lake audit: all 25 directory-level reads, 425 manifest
  hashes, per-dataset uniform schemas, counts, duplicates, first/last timestamps,
  symbol/market identity and timestamp-column agreement passed. Full snapshot:
  [lighter_lake_audit_20261017.json](lighter_lake_audit_20261017.json).
- Headless Chrome: roadmap, CSV pipeline reference and EDA frontend rendered at
  desktop light, desktop dark and mobile light widths (nine render checks), with
  no runtime exceptions or document overflow. Roadmap market switching was
  checked for ETH/BTC/NEAR. Light/dark/mobile roadmap screenshots were inspected;
  `ethpredict-validation-preview.png` was regenerated. The existing pipeline
  page's mobile command-card overflow was fixed with `min-width: 0`.
- `git diff --check HEAD -- . ':!eda_frontend/plotly.min.js'`: passed. The bundled
  upstream Plotly asset retains its original trailing whitespace and license.
- Changed text files were scanned for PEM private keys, GitHub token patterns and
  literal Lighter private-key assignments; no matches. This is a targeted review,
  not a guarantee that every possible sensitive string can be detected.

## Data and issue conclusions

Parquet totals 303,103,519 bytes (303.10 MB; 289.06 MiB). Funding has 31 shared
missing hours per market. Six missing ETH hours exist in an older archive; the
claim that all gaps are exchange-wide unavailable data is not supported. Exact
request-boundary causality and recovery remain [#54](https://github.com/pranay5255/ETHPredict/issues/54).
Schema repair and normal current-month refresh do not recover these old gaps.

No existing open issue's acceptance criteria were completed by backfill alone.
#8 joins is already closed. #16/#17/#20 wait for PR #51's merge; #43–#45 require
recurring snapshots, packaged current inference and a dashboard. No unrelated
issue was closed. GitHub snapshot: 21 open issues, including new #54.

No new research fits, strategy searches, final-test evaluations, data repairs or
signed trading calls were performed for this update. Historical reports and
notebook outputs retain their original evidence scope. The local data lake,
credentials, model artifacts and full run outputs remain outside this commit
series. General training lake ingestion is still future work.
