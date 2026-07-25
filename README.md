# Train RNN ASD

[![CI](https://github.com/bmaged23/train-rnn-asd/actions/workflows/ci.yml/badge.svg)](https://github.com/bmaged23/train-rnn-asd/actions/workflows/ci.yml)

RNN-based Active Speaker Detection (ASD): given a face track, predict per-frame
or per-window whether that person is speaking (`SPEAKING_AUDIBLE`) or not
(`NOT_SPEAKING`), using only facial landmarks — no audio.

A MediaPipe FaceLandmarker extracts a 478-point face mesh per frame; a
bidirectional LSTM consumes the resulting landmark sequence (by default, just
the 40 mouth landmarks + Mouth Aspect Ratio, 121 features/frame) and outputs a
speaking/not-speaking prediction.

## Two model variants

| | Architecture | Prediction granularity | Script family |
|---|---|---|---|
| **Per-frame** | Many-to-many BiLSTM | One label per frame | `scripts/modeling/{train,evaluate}/frames*.py` |
| **Windowed** | Many-to-one BiLSTM | One label per fixed-length window (2s / 50 frames) | `scripts/modeling/{train,evaluate}/windowed*.py` |

The windowed variant is the project's primary focus — it consistently
outperforms the per-frame model on this data (see Results below).

## Datasets

Three datasets share the same processed schema (`combined_landmarks.csv`) and
can be combined for training:

- **UniTalk-ASD** — the base dataset (HuggingFace, `plnguyen2908/UniTalk-ASD`)
- **AVA-ActiveSpeaker** — subset pulled from Google's AVA dataset
- **WASD** (Wilder Active Speaker Detection) — subset pulled via YouTube, since
  WASD's own video archive hits a Google Drive download quota

Each dataset has its own download → feature-extraction → split scripts (see
Pipeline below); training scripts opt into extra datasets via
`extra_sources=[ava_source(split), wasd_source(split)]`.

## Pipeline

Each stage is its own folder under `scripts/`, run in order:

```
scripts/
├── dataset/     download raw video/annotations for a dataset
│   ├── download_dataset.py        UniTalk-ASD (HuggingFace)
│   ├── download_ava_subset.py     AVA-ActiveSpeaker subset
│   ├── download_wasd_subset.py    WASD subset (via YouTube)
│   └── download_model.py          MediaPipe FaceLandmarker model
│
├── features/    crop faces + extract landmarks into combined_landmarks.csv
│   ├── extract_landmarks.py       ad-hoc/single-folder extraction
│   ├── process_dataset.py         full UniTalk-ASD train/val batch run
│   ├── process_ava_dataset.py     same, for the AVA subset
│   └── process_wasd_dataset.py    same, for the WASD subset
│
├── splits/      carve the raw "val" pool into held-out val/test
│   ├── split_val_test.py          UniTalk-ASD
│   ├── split_ava_val_test.py      AVA
│   └── split_wasd_val_test.py     WASD
│
└── modeling/
    ├── train/       train a model from scratch (5 variants — see below)
    ├── evaluate/    evaluate a trained checkpoint on the held-out test split
    └── infer.py     run a trained model on new footage
```

### Train/evaluate variants

Each pairs one architecture (`frames` = per-frame, `windowed` = windowed) with
one data combination, and writes to its own checkpoint/log/eval files so runs
never overwrite each other and stay directly comparable:

| Script | Architecture | Data |
|---|---|---|
| `frames.py` | per-frame | UniTalk-ASD only |
| `frames_combined.py` | per-frame | UniTalk-ASD + AVA |
| `windowed.py` | windowed | UniTalk-ASD only |
| `windowed_combined.py` | windowed | UniTalk-ASD + AVA |
| `windowed_all_combined.py` | windowed | UniTalk-ASD + AVA + WASD |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # fill in HF_TOKEN etc. if needed — see comments in the file
```

`requirements.txt` installs the CUDA build of `torch` by default; on a
constrained connection, `pip install --index-url https://download.pytorch.org/whl/cpu torch`
gets a much smaller CPU-only build instead (fine for dev/smoke-testing, not
real training runs).

## Usage

```bash
# 1. Download a dataset
python scripts/dataset/download_dataset.py

# 2. Extract landmarks
python scripts/features/process_dataset.py

# 3. Split the held-out val pool into val/test
python scripts/splits/split_val_test.py

# 4. Train
python scripts/modeling/train/windowed.py

# 5. Evaluate on the held-out test split
python scripts/modeling/evaluate/windowed.py --checkpoint best
```

Swap `dataset`/`process_dataset`/`split_val_test` for their `_ava`/`_wasd`
counterparts to bring in the other datasets, and use a `_combined`/
`_all_combined` train/evaluate script to train on more than one dataset at
once.

## Results

Windowed model, trained from scratch on UniTalk-ASD + AVA + WASD combined,
evaluated on the held-out combined test split (9,533 windows):

| Metric | Value |
|---|---|
| Accuracy | 86.84% |
| Precision | 76.09% |
| Recall | 82.39% |
| F1 | 79.11% |
| ROC-AUC | 93.88% |
| PR-AUC | 88.67% |

## Project structure

```
config.py           Project-wide paths and constants (single source of truth)
src/                 Shared library code (datasets, models, metrics, training utils)
scripts/             Pipeline entry points (see Pipeline above)
data/                Downloaded/processed data (gitignored)
checkpoints/         Trained model weights (gitignored)
logs/                Training curves/metrics (gitignored)
evaluation*/         Held-out test-set evaluation results (gitignored)
notebooks/           Exploratory analysis
```
