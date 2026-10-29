# Pinned checkpoint provenance — verified before freeze

The September 15, 2025 TimesFM 2.5 family announcement does not establish the
publication date of the exact snapshot used by this grader. Primary-source
verification on October 29, 2026 found:

- The [October 1 weight/config refresh](https://huggingface.co/google/timesfm-2.5-200m-pytorch/commit/d418f3e8a8fa79d655b391c158f0ee8d68fe68c9)
  introduced the currently pinned safetensors bytes, SHA-256
  `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`,
  size 925,181,104 bytes. These match the existing local registry and the
  immutable copies used by the baseline worker; no weights were upgraded.
- The pinned [October 2 revision](https://huggingface.co/google/timesfm-2.5-200m-pytorch/commit/1d952420fba87f3c6dee4f240de0f1a0fbc790e3)
  documents a QKV layout change for execution speed. Its model card retains
  Apache-2.0. This is the revision already selected by the grader.
- The public model API reports snapshot timestamp
  `2025-10-02T17:21:38.000Z`. File metadata reports the weight/config refresh at
  `2025-10-01T22:16:35.000Z`. The repository's September 2 creation date alone
  proves neither the family release nor this snapshot's publication.

The corrected registry uses **October 2, 2025** as the conservative pinned-snapshot
release date, and **October 3, 2025 00:00 UTC** as the first eligible evidence
boundary. This supersedes the earlier registry's family-announcement cutoff.
Periods before that boundary must remain “possibly in pretraining.” The
specified August 2026 holdout still starts after the snapshot's release; no
holdout evaluation is authorized or performed by this correction.

The completed bounded baseline measurements used the same pinned weight bytes,
training, forecasts, positions and accounting. Their report/cache metadata was
corrected after the measurement process finished. The
[correction receipt](grader_baseline_metadata_correction.json) verifies identical
numeric-score/resource digests, strategy-source hashes and artifact manifests.
A second full measurement under the final independent causality guard used the
corrected registry from the outset. Its [validation receipt](grader_baseline_final_validation.json)
confirms identical performance and all 60 position files. The
[baseline table](grader_v1_baselines.md) contains this final repeat and its runtimes. All
rubric numbers, anchors, weights, costs and gates were retained; only the
calibration state changed to complete. The best net baseline is flat with a
pre-gate score of 0.100, and all six pre-gate scores lie between 0.100 and 0.174.
Every final baseline score remains zero.
Publication timing alone does not establish the complete pretraining corpus
or the statistical independence of any evaluation period.
