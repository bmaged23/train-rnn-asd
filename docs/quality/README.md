# Reliability work — in progress

This first stage adds executable evaluation contracts and split checks. It does
not reproduce the historical model scores or qualify a trained model/container.
No trained checkpoint or processed landmark dataset is included in this checkout;
real model qualification must wait for those artifacts.

## Reproduce the offline checks

Python 3.13.16 on Linux x86-64 is the tested quality-check environment:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/ci.txt
python -m pip check
MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.cache/matplotlib" python -m unittest discover -s tests -v
ruff check config.py src scripts tests
```

`requirements/evaluation.txt` is the exact data/metrics environment. `requirements/cpu.txt` adds CPU-only Torch and its exact dependencies; the CI
include adds pinned Ruff and YAML-validation tools. MediaPipe and image/VLM training stacks
remain separate. These are not full GPU/application locks.
The old root requirements remain provisional until model-family environments
are tested separately. Hosted CI now runs these checks rather than only syntax.

## Metric contracts

- ROC-AUC is JSON `null` if the true labels contain only one class. No-positive
  average precision is also `null`; all-positive AP is mathematically 1 but
  does not establish two-class discrimination.
- Reports label undefined metrics explicitly. Plots explain missing classes
  instead of drawing undefined curves or dividing absent-class rows by zero.
- Invalid shapes, empty/mismatched arrays, nonbinary labels/predictions and
  invalid/nonfinite probabilities fail before computing a misleading score.
- Confusion matrix, accuracy, precision, recall, F1, ROC-AUC and average precision
  retain their prior meaning on normal two-class inputs. Additional outputs are
  specificity, balanced accuracy, MCC, Brier score and class support.
- `pr_auc` means **average precision**, not trapezoidal integration of a PR curve.
  Probability thresholds must be chosen using validation data, not test data.

The tests use hand-checked synthetic arrays plus real CPU Torch, Pandas/PyArrow,
state-dictionary loading and plotting. Their success is software evidence,
not evidence that a trained speaker detector is accurate.

## RNN, provenance and cache contracts

The per-frame and windowed LSTMs now import through the `src` package without
requiring a training script to add `src/` to Python's search path. Tests run
actual CPU LSTMs with random test weights: output shapes, valid-prefix padding
invariance, unsorted batch lengths, individual/batch agreement and exact
state-dictionary round trips. These are not trained-model evaluations.

Windowed training caches now include SHA256 of each processed CSV, ordered feature columns, geometry
pairs and source order. Changing labels/features invalidates the cache even when file size and
mtime are unchanged. Existing older caches are not selected. Hashing is a
streaming pass over the source files and adds I/O; original CSVs must remain
available to verify cache identity. Atomic writes prevent interrupted saves
from publishing partially written pickle files. Cached files are local/trusted
artifacts, not a safe interchange format for untrusted uploads.

Landmark per-source evaluation uses the native `source_label` in each batch,
as the image/VLM evaluators already do. It no longer reloads all source CSVs
to infer a map from track IDs. Reused IDs in different corpora retain the
correct source; missing/mismatched provenance and empty evaluations fail
explicitly instead of silently dropping or misattributing rows.

The RNNs are bidirectional and consume completed tracks/windows. These changes
do not turn them into causal livestream models.

## Audit real processed splits

```sh
python scripts/validation/audit_split_isolation.py \
  --source unitalk \
    data/processed/train/combined_landmarks.csv \
    data/processed/val/val_split.csv \
    data/processed/val/test_split.csv \
  --hash-inputs --output output/audits/unitalk-isolation.json
```

Repeat `--source NAME TRAIN_CSV VAL_CSV TEST_CSV` for other datasets. The audit
streams metadata, never loads landmark tensors, and never rewrites the splits.
It records counts, label coverage, optional full SHA256 fingerprints and overlap
examples. Exit code 1 means review is required: shared track/recording IDs,
missing recording identifiers, empty data or invalid metadata. Review invalid
label rows against the actual model's eligibility/masking policy.

**Disjoint tracks alone do not prove recording isolation.** Different people
from one source video can otherwise appear in both validation and test sets.
This tool compares exact identifiers within each source. It cannot establish
unseen-subject generalization, normalize differently named source films, or
prove duplicate-image absence across corpora. Its checks do not retroactively
prove that historical results were leaked; the actual datasets must be audited.

## Laptop-safe concurrency

Historical defaults included 64 extraction processes and 32 image-loader
workers. Each pool now respects `ASD_WORKER_LIMIT`, which defaults to at most
four and leaves a CPU available where possible. Existing lower limits remain.
For a suitable server, explicitly set `ASD_WORKER_LIMIT=64` to restore historical
pool capacities. This is a per-pool cap, not a cap on all processes combined,
and it does not make VLM/image batches fit smaller GPU memory.

## Remaining release work

Locate trained checkpoints, matching feature/config metadata and processed
train/validation/test data; verify split isolation and real held-out results.
Then benchmark actual models/flows on CPU/GPU, establish qualified serving and
Docker assets/terms, test a real demo/API, and gate public image delivery on
real model results. No random-weight or synthetic-only accuracy may substitute
for those stages. Keep large six-source downloads and VLM installation opt-in.
