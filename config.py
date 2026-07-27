"""Project-wide paths and constants, filled in incrementally as pipeline stages are built."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- HuggingFace dataset ---
HF_DATASET_REPO_ID = "plnguyen2908/UniTalk-ASD"
HF_REPO_TYPE = "dataset"
SPLITS = ("train", "val")

# --- Raw data (untouched download) ---
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV_DIR = DATA_RAW_DIR / "csv"
RAW_CLIPS_VIDEOS_DIR = DATA_RAW_DIR / "clips_videos"
RAW_CLIPS_AUDIOS_DIR = DATA_RAW_DIR / "clips_audios"

# --- Processed data ---
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEQUENCES_DIR = DATA_PROCESSED_DIR / "sequences"
FAILED_EXTRACTIONS_FILE = DATA_PROCESSED_DIR / "failed_extractions.txt"

# process_dataset.py output — full train/val batch run, lives under data/processed/
# alongside the other processed-data artifacts.
PROCESSED_DATASET_DIR = DATA_PROCESSED_DIR

# extract_landmarks.py output — kept at project root (outside data/) since it's
# the single-folder ad-hoc/smoke-test output, not the real dataset run.
LANDMARKS_DIR = PROJECT_ROOT / "landmarks"

# --- Per-video CSV column schema (files ship without a header row) ---
CSV_COLUMNS = [
    "video_id", "frame_timestamp",
    "entity_box_x1", "entity_box_y1",
    "entity_box_x2", "entity_box_y2",
    "label", "entity_id",
    "label_id", "instance_id",
]

# --- MediaPipe Face Landmarker (Tasks API) ---
# Lives inside the project (models/face_landmarker.task) so the project is
# self-contained — this used to be a symlink at /data/bola/models/ pointing
# into another user's directory (/data/Mona/attention/); scripts/dataset/download_model.py
# fetches this from the official MediaPipe model bucket if it's ever missing,
# and PersonLandmarkExtractor (extract_landmarks.py) auto-downloads it on
# first use rather than failing.
MODELS_DIR = PROJECT_ROOT / "models"
FACE_LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
NUM_FACE_LANDMARKS = 478  # attention-mesh model output, includes iris landmarks

# Flat index set covering MediaPipe's FACE_LANDMARKS_LIPS connections
# (outer + inner lip contour). Derived from
# mediapipe.tasks.python.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS.
MOUTH_LANDMARK_INDICES = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80,
    81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
    185, 191, 267, 269, 270, 291, 308, 310, 311, 312,
    314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
]

MEDIAPIPE_NUM_FACES = 1
MEDIAPIPE_MIN_FACE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# "CPU" or "GPU" (BaseOptions.Delegate). Benchmarked on this server's H100
# (shared with other tenants' jobs, GPU already at ~100% util / 74-81GB VRAM
# used at benchmark time): a single GPU-delegate process did 61.6 img/s vs.
# 16.9 img/s for a single CPU process, and 4 concurrent GPU processes hit
# 127.3 img/s — already beating the 105-process CPU pool (~100-110 img/s)
# while using a fraction of the CPU cores. Diminishing returns set in by x4
# (x1->x2 was +76%, x2->x4 only +17%), consistent with GPU contention from
# other tenants, so more GPU processes isn't assumed to keep scaling.
MEDIAPIPE_DELEGATE = "CPU"

# Mouth Aspect Ratio: mean(vertical inner-lip gap distances) / horizontal mouth width.
# Index pairs are landmark indices into the 478-point face mesh (same indexing as
# MOUTH_LANDMARK_INDICES / FACE_LANDMARKS_LIPS).
MAR_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]
MAR_HORIZONTAL_PAIR = (61, 291)

# --- landmark normalization (optional, opt-in via PersonLandmarkExtractor(normalize=...)) ---
# Centers landmarks on the nose tip and scales by interocular distance (outer eye corners),
# making them invariant to face size/position within the crop. Off by default so extraction
# stays raw pixel-space unless explicitly requested.
NORMALIZE_LANDMARKS = False
NORMALIZATION_CENTER_INDEX = 1        # nose tip
NORMALIZATION_SCALE_PAIR = (33, 263)  # outer eye corners (interocular distance)

# --- extract_landmarks.py output ---
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
LANDMARKS_JSON_FILENAME = "landmarks.json"
SAVE_LANDMARKS_JSON = True  # process_dataset.py sets this False — the combined CSV is enough
ANNOTATED_SUBDIR_NAME = "annotated"
ANNOTATION_POINT_RADIUS = 1
ANNOTATION_MESH_COLOR = (0, 255, 0)
ANNOTATION_MOUTH_COLOR = (0, 255, 255)
JSON_INDENT = 2

# --- process_dataset.py output ---
# One row per frame, appended as each person finishes. CSV, not JSON: same data,
# but directly loadable as a numeric array for training with no parsing step, and
# meaningfully smaller on disk (no repeated key names / bracket punctuation per frame).
COMBINED_LANDMARKS_FILENAME = "combined_landmarks.csv"
# One person_id per line, written only after that person's CSV rows are durably
# flushed+fsynced — the actual resume marker (see DatasetProcessor._load_completed_ids).
# Kept separate from the CSV itself since a crash mid-row-write wouldn't be
# detectable by scanning the CSV's person_id column the way a broken JSON line was.
COMPLETED_IDS_FILENAME = "completed_ids.txt"
# Landmark coordinates and mar are written at fixed decimal precision rather than
# Python's full float repr (~17 significant digits) — 6 decimals is ~1e-6 in
# normalized (interocular-distance) units, far finer than MediaPipe's actual
# detection accuracy, but shrinks each field from ~19-20 chars to ~9.
CSV_FLOAT_DECIMALS = 6

# Worker processes for parallel extraction. With MEDIAPIPE_DELEGATE="GPU", 4
# concurrent processes measured faster in aggregate than the entire 105-process
# CPU pool (see MEDIAPIPE_DELEGATE comment) — going higher wasn't beneficial in
# that benchmark (contention from other tenants sharing the GPU) and risks the
# GPU's limited free VRAM under load from other users' jobs. If MEDIAPIPE_DELEGATE
# is switched back to "CPU", raise this back toward the core count (112 on this
# server) instead.
NUM_EXTRACTION_WORKERS = 64

# Optional: folder to run scripts/features/extract_landmarks.py against directly (train/val
# person folder or an arbitrary eval folder of images). Falls back to a sample
# train folder if unset.
EXTRACT_LANDMARKS_PERSON_DIR = os.getenv("EXTRACT_LANDMARKS_PERSON_DIR") or None

# --- src/dataset.py (train/val DataLoader) ---
# Per-track (person_id) detected-frame ratio threshold: tracks with fewer than
# this fraction of frames having a detected face are dropped from the dataset
# index entirely (not deleted from disk/CSV — just excluded from what
# LandmarkSequenceDataset loads). ~53% of frames overall are detected and
# ~53% of tracks fall below a 50% per-track detection rate (see
# notebooks/03_explore_processed_dataset.ipynb), so this threshold is the
# main lever on how much of the dataset actually gets trained on.
MIN_DETECTED_FRAMES_RATIO = 0.5

# Batch size for both train and val DataLoaders (src/dataset.py).
BATCH_SIZE = 32

# If True, batches are built so each one's SPEAKING_AUDIBLE frame percentage
# tracks the dataset's overall percentage (~38% on train, measured 2026-07-16)
# instead of swinging batch-to-batch under plain random sampling (empirically
# ranged 14%-75% per batch, stdev ~9.9pp, around that same ~38% average).
# NOT 50/50 balancing, and NOT oversampling/undersampling — every track is
# still used exactly once per epoch, so the class prior the model actually
# trains on is unchanged; only which tracks land in the same batch changes.
# See src/dataset.py's StratifiedBatchSampler.
BALANCED_BATCHES = True

# Feature-selection toggles — apply identically to train/val/test (single
# source of truth here, not a per-call override) since the model's
# input_size is derived from these and a checkpoint trained with one
# combination won't load into a model built with another.
# True: only the 40 MOUTH_LANDMARK_INDICES points (x,y,z). False: all
# NUM_FACE_LANDMARKS (478) points — ~11.8x more landmark columns, so
# noticeably more memory and slower CSV loads (see src/dataset.py).
USE_MOUTH_LANDMARKS_ONLY = True
# Whether to append the MAR (mouth aspect ratio) scalar feature.
USE_MAR = True

# --- scripts/splits/split_val_test.py ---
# Splits data/processed/val/combined_landmarks.csv (by whole person_id track,
# never splitting a track across files) into two physical CSVs: "val" (used
# every epoch for early stopping / model selection — indirectly influences
# training) and a held-out "test" set (never looked at until final
# evaluation — the actual check for overfitting). Stratified on (a) whether
# the track's detected-frame ratio already clears MIN_DETECTED_FRAMES_RATIO
# and (b) whether the track is SPEAKING_AUDIBLE-majority or NOT_SPEAKING-
# majority, so both output files carry a similar detection-quality and class
# balance rather than an arbitrary random split. Deterministic via
# VAL_TEST_SPLIT_SEED — reruns produce the same assignment.
VAL_SPLIT_RATIO = 0.7  # fraction of val tracks kept in val_split.csv; remainder -> test_split.csv
VAL_TEST_SPLIT_SEED = 42
VAL_SPLIT_FILENAME = "val_split.csv"
TEST_SPLIT_FILENAME = "test_split.csv"

# --- src/model.py (RNN model) ---
# Bidirectional LSTM: hidden_size per direction (the LSTM's actual per-timestep
# output is 2x this, forward+backward concatenated), stacked layers, and
# inter-layer dropout (only applied when NUM_RNN_LAYERS > 1 — nn.LSTM ignores
# dropout on a single-layer stack).
HIDDEN_SIZE = 256
NUM_RNN_LAYERS = 3
DROPOUT = 0.3

# torch.set_num_threads() for CPU runs. This box has 64 cores and PyTorch
# defaults to using all of them, which for a model this small (hidden_size=128)
# causes thread-contention overhead to dominate over actual compute — measured
# a single forward pass dropping from several minutes to ~0.1s after capping
# threads. Irrelevant once training moves to GPU, but keep it applied for any
# CPU run (smoke tests, CPU-only training/eval) to avoid this trap.
TORCH_CPU_THREADS = 4

# --- scripts/modeling/train/frames.py ---
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
TRAIN_METRICS_FILENAME = "train_metrics.csv"  # one row per epoch, gitignored (logs/*.csv)
TRAIN_LOG_FILENAME = "train.log"              # human-readable run log, gitignored (logs/*.log)
BEST_CHECKPOINT_FILENAME = "best_model.pt"    # lowest val loss so far, gitignored (checkpoints/*.pt)
LAST_CHECKPOINT_FILENAME = "last_model.pt"    # final epoch, regardless of val loss

# High ceiling since early stopping (on "val" loss) is what actually decides
# when training stops — NUM_EPOCHS just needs to be comfortably above where
# that's expected to trigger, not a target to hit.
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
GRADIENT_CLIP_NORM = 5.0  # nn.utils.clip_grad_norm_ max_norm — LSTMs are prone to exploding gradients

# Early stopping: stop once "val" loss hasn't improved for this many
# consecutive epochs. "val" here is the 70% split from split_val_test.py —
# "test" (the other 30%) is never touched during training, only for a final
# evaluation after stopping. Lowered 100->20 (2026-07-18, explicit user
# request) — checkpoint selection already keeps the best-val-loss epoch
# regardless of how much longer training continues afterward, so a high
# patience only ever cost wasted compute in every run this project has
# seen (flagged repeatedly, e.g. the full-AVA combined run continued to
# epoch 118 with its actual best at a much earlier epoch) and never
# changed which checkpoint got selected.
EARLY_STOPPING_PATIENCE = 20

# --- scripts/modeling/evaluate/frames.py & src/metrics.py ---
# target_index label convention, shared with src/dataset.py's collate_sequences
# ("labels" comment) and used to order confusion-matrix axes / classification
# report rows consistently everywhere.
CLASS_NAMES = ["NOT_SPEAKING", "SPEAKING_AUDIBLE"]

# scripts/modeling/evaluate/frames.py is the dedicated classification-metrics check (accuracy/
# precision/recall/F1/confusion matrix/ROC-AUC/PR-AUC) — hardcoded to the
# held-out "test" split (see its module docstring) and never run from
# scripts/modeling/train/frames.py. Its own output folder, separate from checkpoints/logs/.
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVAL_METRICS_FILENAME = "eval_metrics.json"               # gitignored (evaluation/*.json)
EVAL_LOG_FILENAME = "eval.log"                             # gitignored (evaluation/*.log)
EVAL_CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"   # gitignored (evaluation/*.png)
EVAL_ROC_CURVE_FILENAME = "roc_curve.png"                  # gitignored (evaluation/*.png)
EVAL_PR_CURVE_FILENAME = "pr_curve.png"                    # gitignored (evaluation/*.png)

# --- scripts/modeling/train/frames.py: post-training progress charts + report (added 2026-07-16) ---
# Training-progress check only — train vs val loss/accuracy across every
# epoch of the run just finished, from logs/train_metrics.csv. Deliberately
# does NOT touch the "test" split (that's scripts/modeling/evaluate/frames.py's job, kept
# separate — see EVALUATION_DIR above). Lives in LOGS_DIR next to
# train_metrics.csv/train.log.
LOSS_CURVE_FILENAME = "loss_curve.png"          # gitignored (logs/*.png)
ACCURACY_CURVE_FILENAME = "accuracy_curve.png"  # gitignored (logs/*.png)
TRAINING_REPORT_FILENAME = "README.md"          # gitignored (logs/*.md)

# --- src/windowed_dataset.py / src/windowed_model.py / scripts/modeling/train/windowed.py /
#     scripts/modeling/evaluate/windowed.py (added 2026-07-16) ---
# Windowed many-to-one variant, alongside (not replacing) the per-frame
# many-to-many pipeline above — see notebooks/… discussion and
# [[training-runs-log]]-style memory for why: our per-frame model predicts
# every frame independently, which is a harder task than most reference
# implementations (e.g. sachinsdate/lip-movement-net) attempt — they classify
# a whole short window as speaking/not-speaking, one label per window, not
# per frame. This tests that same coarser-granularity idea on our own data
# rather than switching problems entirely.
#
# DATASET_FPS=25 is our data's confirmed frame spacing (median frame-to-frame
# timestamp delta = 0.04s, verified 2026-07-16) — WINDOW_SIZE is expressed in
# seconds (WINDOW_SECONDS) and derived from this, not hardcoded in frames, so
# changing the window duration is a one-line edit. History: 1s (25 frames,
# matching lip-movement-net, F1=0.6619) -> 2s (50 frames, F1=0.6803, project's
# best as of 2026-07-17) -> 3s (75 frames, F1=0.6652 without padding, 0.6749
# with WINDOW_PAD_MIN_LENGTH_RATIO=0.5 padding — still short of 2s) -> back
# to 2s now, to test padding at the window size that actually won. A
# 5-second window was checked and rejected: it would drop 80%+ of tracks as
# too short (only 16-19% survive); most 2s tracks already clear a 50%
# padding floor given the milder length requirement (~55-62% survive
# unpadded already), so padding may matter less here than it did at 3s —
# that's exactly what this run is testing.
# WINDOW_STRIDE=WINDOW_SIZE means non-overlapping windows (no frame reused
# across windows within a track) — this is what val/test always use, so
# early-stopping/evaluation signal stays honest (heavily overlapping val/test
# windows would be mostly duplicates of each other, inflating apparent
# sample count with correlated, non-independent examples).
DATASET_FPS = 25
WINDOW_SECONDS = 2
WINDOW_SIZE = WINDOW_SECONDS * DATASET_FPS
WINDOW_STRIDE = WINDOW_SIZE

# Train-only overlapping-window stride (added 2026-07-16, per explicit user
# request) — ~50% overlap (half of WINDOW_SIZE) multiplies the number of
# training windows drawn from the same tracks, directly targeting the "only
# ~3 windows per track" limited-diversity theory (see [[windowed-approach]]
# in project memory) without touching val/test's non-overlapping WINDOW_STRIDE
# above. Deliberately train-only: overlapping windows share most of their
# frames, so using them for early stopping/evaluation would make those
# signals look smoother/more stable than they really are.
WINDOW_TRAIN_STRIDE = WINDOW_SIZE // 2

# Whether tracks shorter than WINDOW_SIZE get ONE zero-padded window each
# (using all their real frames, model given the true length via
# pack_padded_sequence) instead of being dropped entirely. Added 2026-07-17,
# toggleable per explicit user request. Hypothesis being tested: the recall
# dip observed going from WINDOW_SECONDS=2 to =3 (F1 0.6803->0.6652 despite
# val_loss/precision/ROC-AUC all improving) is a data-scarcity artifact —
# 68% of tracks are too short to even form one 3s window — not a
# fundamental weakness of longer windows; padding recovers that lost data.
# Applied identically to train/val/test (unlike WINDOW_TRAIN_STRIDE above) —
# padding doesn't create correlated/duplicate samples the way overlap does,
# each padded window is still exactly one genuine, distinct track, so there's
# no honesty concern for val/test here.
#
# First real test (WINDOW_SIZE=75, no minimum-length floor — see
# WINDOW_PAD_MIN_LENGTH_RATIO below) came out WORSE across nearly every
# metric (val_loss 0.267->0.362, precision 0.744->0.673, F1 0.6652->0.6463)
# — plausible cause: tracks as short as 21 frames (28% real content in a
# 75-frame window) were being padded and used, adding noisy, mostly-padding
# samples rather than genuine extra data. Adding a WINDOW_PAD_MIN_LENGTH_RATIO
# floor recovered most of the loss at 3s (F1 0.6463->0.6749), but the same
# floor HURT at 2s (F1 0.6803->0.6466, WINDOW_SECONDS=2) — most 2s tracks
# already clear the length bar unpadded, so padding there disproportionately
# adds the shortest/noisiest tracks instead of recovering genuinely lost
# data. Set back to False here to reproduce the project's best-known
# result (2s windows, no padding, F1=0.6803).
WINDOW_PAD_SHORT_TRACKS = False

# Minimum fraction of WINDOW_SIZE a short track's real length must reach to
# be padded at all, when WINDOW_PAD_SHORT_TRACKS is True — tracks shorter
# than this are still dropped, same as when padding is off entirely. Added
# 2026-07-17 after the first padding test (no floor) hurt results — a track
# that's mostly padding (e.g. 28% real content) is plausibly noise, not
# useful extra data. 0.5 means a track must supply at least half the
# window's real frames to qualify.
WINDOW_PAD_MIN_LENGTH_RATIO = 0.5

# Windows are built from already-track-filtered data (same MIN_DETECTED_FRAMES_RATIO-
# passing tracks as the per-frame pipeline), but an individual 25-frame window
# can still land in a low-detection stretch of an otherwise-fine track — reuse
# the same threshold at window granularity: windows below this detected-frame
# ratio are dropped, not label-guessed.
WINDOW_MIN_DETECTED_FRAMES_RATIO = MIN_DETECTED_FRAMES_RATIO

# The windowed model overfits faster/harder than the per-frame one despite
# identical LSTM capacity — plausible cause: 50K windows come from only ~17K
# tracks (~3 windows/track), less effective person/scene diversity than the
# window count suggests, over a much shorter (25-frame) input. AdamW +
# weight_decay was tried first (2026-07-16) and swept across 4 values
# (0, 1e-3, 3e-3, 1e-2) with a fixed seed — none beat plain Adam with no
# decay at all, so that approach was dropped entirely (removed from
# scripts/modeling/train/windowed.py, back to plain torch.optim.Adam) — see
# [[windowed-approach]] for the sweep table. Reduced capacity (32/1) was
# tried next — it genuinely fixed the overfitting *shape* (final gap
# 16pp->7pp) but gave the *worst* F1 of everything tried (0.6241), since the
# best-checkpoint selection already protected the reported metric from
# overfitting — the real bottleneck is best-checkpoint quality, not how much
# it degrades afterward. Reverted back to matching the per-frame model's
# HIDDEN_SIZE=128/NUM_RNN_LAYERS=2 so the current lever (overlapping
# windows, see WINDOW_TRAIN_STRIDE above) is tested in isolation against the
# best-known baseline (F1=0.6511, full capacity, seeded, no weight decay),
# not compounded with an untested capacity change.
WINDOWED_HIDDEN_SIZE = 256
WINDOWED_NUM_RNN_LAYERS = 2

# Fixed seed for reproducible scripts/modeling/train/windowed.py comparisons — added
# 2026-07-16 after noticing run-to-run F1 swings that couldn't be
# distinguished from real config effects without one (no seed existed
# before this). Deliberately scoped to the windowed pipeline only —
# scripts/modeling/train/frames.py (per-frame) is untouched, consistent with the
# "don't touch the per-frame pipeline" structural decision.
WINDOWED_SEED = 42

# Own checkpoint files — never overwrites the per-frame model's checkpoints.
WINDOWED_BEST_CHECKPOINT_FILENAME = "windowed_best_model.pt"  # gitignored (checkpoints/*.pt)
WINDOWED_LAST_CHECKPOINT_FILENAME = "windowed_last_model.pt"  # gitignored (checkpoints/*.pt)

# Own train-progress artifacts, same LOGS_DIR, "windowed_" prefix so they
# don't collide with the per-frame run's files.
WINDOWED_TRAIN_METRICS_FILENAME = "windowed_train_metrics.csv"    # gitignored (logs/*.csv)
WINDOWED_TRAIN_LOG_FILENAME = "windowed_train.log"                # gitignored (logs/*.log)
WINDOWED_LOSS_CURVE_FILENAME = "windowed_loss_curve.png"          # gitignored (logs/*.png)
WINDOWED_ACCURACY_CURVE_FILENAME = "windowed_accuracy_curve.png"  # gitignored (logs/*.png)
WINDOWED_TRAINING_REPORT_FILENAME = "windowed_README.md"          # gitignored (logs/*.md)

# Own evaluation output folder — kept separate from EVALUATION_DIR (per-frame
# results) rather than filename-prefixed, since "one sample" means something
# different here (a window, not a detected frame) and mixing the two in one
# folder invites exactly the kind of confusion already seen with the
# frames-vs-tracks "samples" terminology (see [[dataloader-design]]).
WINDOWED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_windowed"

# --- scripts/dataset/download_ava_subset.py (added 2026-07-17) ---
# AVA-ActiveSpeaker fine-tuning data: a second, independent ASD dataset (CC BY
# 4.0, commercial-friendly) used to fine-tune the windowed pipeline and check
# generalization beyond UniTalk-ASD. Same core annotation schema UniTalk-ASD
# itself follows (video_id, frame_timestamp, entity_box_x1/y1/x2/y2, label,
# entity_id) — face boxes are human-annotated in the source CSV, no detector
# needed to crop. Kept fully separate from data/raw/ and data/processed/
# (never merged into UniTalk's train/val/test) so it stays a distinct,
# removable fine-tuning set rather than silently growing the primary training
# data — mirrors this project's existing pattern of keeping parallel
# pipelines (e.g. windowed vs. per-frame) in separate files/dirs.
AVA_DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw_ava"
AVA_RAW_CSV_DIR = AVA_DATA_RAW_DIR / "csv"
AVA_RAW_CLIPS_VIDEOS_DIR = AVA_DATA_RAW_DIR / "clips_videos"  # no clips_audios/ — audio is never used
AVA_PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed_ava"

AVA_ANNOTATIONS_URL_TRAIN = "https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_train_v1.0.tar.bz2"
AVA_ANNOTATIONS_URL_VAL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_activespeaker_val_v1.0.tar.bz2"
AVA_FILE_NAMES_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt"
AVA_VIDEO_BASE_URL = "https://s3.amazonaws.com/ava-dataset/trainval"

# Total annotated-video hours to pull, split proportionally across AVA's own
# official train/val CSVs (not a new split boundary we invent) — e.g.
# 3.0 * (1 - 0.2) = 2.4h train, 3.0 * 0.2 = 0.6h val. None means no cap at
# all — select every video in both official CSVs (120 train + 33 val, the
# full public AVA-ActiveSpeaker release; the commonly-cited "262 movies"
# figure includes a held-out test set Google never released annotations
# for, so it isn't actually downloadable). This is a plain config constant,
# not a CLI flag, per explicit user request ("we need a physical code to
# change it not just commands") — it's the one thing to edit for a
# bigger/smaller/unlimited future pull. First run (2026-07-17) used 3.0 to
# validate the pipeline before committing to the full ~70GB/~150-video pull;
# now set to None for the real full-dataset run.
AVA_SUBSET_HOURS = None
AVA_SUBSET_VAL_FRACTION = 0.2

# Concurrent ffmpeg extractions in scripts/dataset/download_ava_subset.py — each is
# an independent subprocess (CPU-bound video decode, no GPU decode used) so
# this parallelizes well on this box's 64 cores, unlike NUM_EXTRACTION_WORKERS
# above which is capped low because that step's bottleneck is a shared,
# VRAM-limited GPU delegate pool — no such constraint here.
NUM_AVA_DOWNLOAD_WORKERS = 8

# AVA's 3rd raw label class (mouth visibly moving, audio just not
# clear/audible — official AVA docs use "SPEAKING_BUT_NOT_AUDIBLE"; some
# secondary sources say "SPEAKING_NOT_AUDIBLE" — both are mapped here since
# the exact string isn't confirmed until the real CSV is downloaded and
# validated) is remapped to SPEAKING_AUDIBLE rather than dropped or mapped to
# NOT_SPEAKING — our model is vision-only (mouth landmarks + MAR, no audio),
# so "not audible" carries no signal we can see; treating it as NOT_SPEAKING
# would label real mouth movement as negative (injecting noise), and
# dropping it wastes usable data for no benefit. Decided explicitly with the
# user 2026-07-17 — see [[windowed-approach]]. download_ava_subset.py
# validates every raw label string against this map's keys at runtime and
# fails loudly on an unrecognized value, rather than silently mismapping.
AVA_LABEL_ID_MAP = {
    "NOT_SPEAKING": 0,
    "SPEAKING_AUDIBLE": 1,
    "SPEAKING_AND_AUDIBLE": 1,
    "SPEAKING_NOT_AUDIBLE": 1,
    "SPEAKING_BUT_NOT_AUDIBLE": 1,
}

# --- scripts/dataset/download_wasd_subset.py / scripts/features/process_wasd_dataset.py
#     (added 2026-07-22) ---
# WASD (Wilder Active Speaker Detection, TBIOM 2025, CC-BY-SA-4.0) — a third
# raw source alongside UniTalk-ASD and AVA, added to cover conditions neither
# of those two has much of: surveillance-quality audio/face degradation,
# occlusion, and overlapping/multi-speaker talk. Unlike AVA, WASD's own
# train_orig.csv/val_orig.csv already ship with a header and the exact
# CSV_COLUMNS column names/order (video_id, frame_timestamp, entity_box_x1-y2,
# label, entity_id, label_id, instance_id) and a binary label_id (0/1 —
# confirmed against the real downloaded CSV, no third class present), so no
# label remapping is needed the way AVA_LABEL_ID_MAP was — the CSVs are just
# copied in unmodified. Kept fully separate from data/raw/, data/raw_ava/, and
# their processed/ counterparts (own raw_wasd/ + processed_wasd/), mirroring
# the AVA integration's separate-until-explicitly-combined pattern.
WASD_DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw_wasd"
WASD_RAW_CSV_DIR = WASD_DATA_RAW_DIR / "csv"
WASD_RAW_CLIPS_VIDEOS_DIR = WASD_DATA_RAW_DIR / "clips_videos"  # no clips_audios/ — audio is never used
WASD_PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed_wasd"

# Used only as scratch space while fetching/cropping source video — deleted
# once cropping finishes, since only the cropped face JPGs + CSVs are the
# actual dataset artifacts.
WASD_STAGING_DIR = WASD_DATA_RAW_DIR / "_staging"

# Google Drive file ID for WASD's csv.zip (from the WASD repo's own
# prepare_setup.py, github.com/Tiago-Roxo/WASD) — small (~0.3GB) and never
# hit Google's per-file daily download quota, unlike WASD_videos.zip (~27GB)
# below. The maintainers' "Option A" direct-download zips are listed as
# "Coming soon" (checked 2026-07-22), so gdown against this ID is the only
# working path today for the CSVs.
WASD_CSV_GDRIVE_ID = "1calMd83IIzYnvETY3bee7juRACMBuuyR"

# WASD's own WASD_videos.zip (Google Drive, ~27GB, gdrive id
# 1F6pYUNz1u23Q-PvPHpxzm4RUhgFCfVYL) hit Google's "too many users have
# downloaded this file recently" daily quota wall on first attempt
# (2026-07-22) — a hard block on Google's end, not fixable by retrying
# differently, only by waiting ~24h. Source videos turned out to be ordinary
# YouTube uploads (video_id in the CSV is "{youtube_id}_{start}-{end}" —
# confirmed against a real clip: face boxes tracked the correct speaker at
# the correct timestamp when pulled straight from YouTube), so
# download_wasd_subset.py fetches each annotated time range directly via
# yt-dlp instead of waiting on the Drive quota. Some of WASD's 164 source
# videos may have gone private/deleted/re-uploaded (new id) since the 2023
# paper — those are skipped and counted, never guessed at.
# vcodec pinned to avc1 (H.264) explicitly, not just ext=mp4 — YouTube serves
# both avc1 and av01 (AV1) inside mp4 containers at the same resolution, and
# "best" picks by bitrate, not decodability. First real smoke test
# (2026-07-22) picked an av01 stream for one video and cv2.VideoCapture
# failed to decode every single frame from it ("Failed to get pixel format")
# — 0 frames cropped despite the download succeeding. avc1 decoded cleanly in
# the original manual verification clip, so it's pinned first with a same-
# resolution avc1 fallback before ever falling back to "any codec, any res".
YOUTUBE_URL_TEMPLATE = "https://www.youtube.com/watch?v={video_id}"
YOUTUBE_FORMAT = "bestvideo[vcodec^=avc1][height<=720]/bestvideo[vcodec^=avc1]/best[height<=720]/best"
# Padding added past each clip's annotated end second so the last labeled
# frame_timestamp (subclip-relative, can land right at the nominal end) is
# still inside the downloaded range — same rationale as AVA's extract_frames
# "+1.0s pad" (scripts/dataset/download_ava_subset.py).
YOUTUBE_CLIP_END_PAD_SECONDS = 1
# Per source-video yt-dlp invocation (all of that video's subclips are
# requested in one call via repeated --download-sections, so this covers up
# to WASD's observed max of 50 subclips for one source video).
YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS = 3600

# Cap on how many source (pre-split) YouTube videos to pull into the final
# dataset — None means every source video referenced by the CSVs (164).
WASD_SUBSET_VIDEOS = None

# Concurrent source-video workers in scripts/dataset/download_wasd_subset.py — each
# worker runs one yt-dlp process (fetching every subclip for that source
# video in a single call) then crops every annotated frame from the results.
# Kept lower than NUM_AVA_DOWNLOAD_WORKERS: this is bound by YouTube-side
# rate limiting/throttling risk on concurrent connections, not local CPU/
# disk, so more workers doesn't reliably mean more throughput.
NUM_WASD_DOWNLOAD_WORKERS = 8

# --- scripts/dataset/download_columbia_subset.py (added 2026-07-25) ---
# Columbia Active Speaker Detection dataset (Chakravarty & Tuytelaars, ACM
# ICMI 2016) — a fourth raw source, small (one 87-minute panel discussion,
# 6 named speakers, ~149K total labeled frames) and not intended as bulk
# training volume the way AVA/WASD are; useful instead as an extra
# generalization/held-out check. No official public dataset page could be
# found despite extensive search (2026-07-25) — the video + ground-truth
# label archive used here is the exact pair github.com/Junhua-Liao/LR-ASD's
# Columbia_test.py already relies on for its own Columbia benchmark
# evaluation, both independently verified working before writing the
# download script: the YouTube video is public, and the Drive-hosted
# col_labels.tar.gz (985KB) downloads and extracts cleanly via gdown,
# containing genuine ground truth crediting the original paper.
#
# Ground truth ships very differently from AVA/WASD's already-normalized
# [0,1] box coordinates — per-speaker txt files (framenum, TLx, TLy,
# square-box-size, speak-flag 0/1) in PIXEL coordinates at the video's
# native frame rate. download_columbia_subset.py converts both:
# frame_timestamp = framenum / native_fps and box coordinates normalized by
# width/height, using the actual downloaded video's probed fps/resolution —
# NOT assumed to be config.DATASET_FPS, no reason this source video happens
# to match that.
#
# Only one video exists, so there's no video-selection/subset-hours concept
# the way AVA/WASD have. Instead, the 6 named speakers (each one single,
# long, continuous track — a much coarser unit than AVA/WASD/UniTalk's many
# short per-scene tracks) are split whole between the "train" and "val" raw
# splits, fixed-seed shuffled — same track-level, no-leakage principle used
# everywhere else in this project (a track's frames never cross a split
# boundary), just applied at a much smaller N. COLUMBIA_NUM_VAL_SPEAKERS
# controls how many go to "val" (which then further divides into
# val_split/test_split via scripts/splits/split_columbia_val_test.py,
# exactly like AVA/WASD); the rest go to "train".
COLUMBIA_DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw_columbia"
COLUMBIA_RAW_CSV_DIR = COLUMBIA_DATA_RAW_DIR / "csv"
COLUMBIA_RAW_CLIPS_VIDEOS_DIR = COLUMBIA_DATA_RAW_DIR / "clips_videos"  # no clips_audios/ — audio is never used
COLUMBIA_PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed_columbia"

COLUMBIA_VIDEO_YOUTUBE_ID = "6GzxbrO0DHM"  # "Wikileaks and Academia - a Panel Discussion at Columbia SIPA"
# col_labels.tar.gz — Drive file id verified working 2026-07-25, sourced
# from github.com/Junhua-Liao/LR-ASD/blob/main/Columbia_test.py.
COLUMBIA_LABELS_GDRIVE_ID = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
# The 6 named panelists' fusion/<name>.txt files ship the complete per-
# speaker ground truth already merged across every tracks_*/ frame-range
# subfolder (confirmed: fusion/bell.txt's row count exactly equals the sum
# of every tracks_*/bell.txt) — read fusion/ directly rather than the
# subfolders individually.
COLUMBIA_SPEAKER_NAMES = ["bell", "boll", "lieb", "long", "sick", "abbas"]
# 2 of 6 speakers (fixed-seed shuffled) go to the raw "val" split, the
# remaining 4 to "train" — see comment above. With only 6 units total this
# is necessarily coarse (the 2-speaker val pool then splits ~1/1 into
# val_split/test_split), an inherent limit of this dataset's size, not a
# tuning knob expected to change often.
COLUMBIA_NUM_VAL_SPEAKERS = 2

# The label txt files' pixel coordinates (TLx, TLy, size) were annotated
# against a video at HALF the resolution of the current 1280x720 YouTube
# upload — not documented anywhere, root-caused by reading
# github.com/Junhua-Liao/LR-ASD's own Columbia evaluation code, which halves
# ITS OWN face-detector coordinates (computed against this same 1280x720
# video) before comparing them to the raw label boxes, i.e. the label boxes
# are implicitly half-scale. Verified directly (2026-07-25): cropping one
# real row's raw box landed on a corner logo/watermark, not a face; cropping
# the same box scaled up by this factor landed exactly on the correct named
# speaker's face, mouth visibly open on a speak=1 row. Applied to TLx, TLy,
# and size uniformly before computing both the pixel crop and the CSV's
# normalized entity_box_* coordinates.
COLUMBIA_LABEL_SCALE_FACTOR = 2
# One cv2.VideoCapture per speaker, opened concurrently against the same
# downloaded video file — safe (independent capture handles) and matches
# the natural per-speaker parallelism unit, no reason to cap below the
# fixed speaker count.
NUM_COLUMBIA_CROP_WORKERS = len(COLUMBIA_SPEAKER_NAMES)

# --- scripts/dataset/download_avspeech_subset.py (added 2026-07-26) ---
# AVSpeech (Google Research, "Looking to Listen at the Cocktail Party") — a
# fifth raw source, added specifically to grow the SPEAKING class, which is
# the minority across the other four sources combined (~30% speaking / ~70%
# not-speaking, see project memory). Every AVSpeech segment is curated by
# its own authors to guarantee the visible person is the sole active speaker
# for the segment's entire 3-10s duration, so every extracted frame gets
# label_id=1 (SPEAKING_AUDIBLE) unconditionally — this source never
# contributes NOT_SPEAKING rows, by design.
#
# Unlike UniTalk-ASD/AVA/WASD/Columbia, AVSpeech does NOT ship a per-frame
# (or even per-clip) face bounding box — its CSV gives only
# (youtube_id, start_sec, end_sec, x_norm, y_norm), a single face-CENTER
# hint at the segment's start frame. This is the first source needing this
# project's own face detection (see RETINAFACE_GPU_ID below) rather than
# cropping an already-provided box.
#
# The documented official download link (storage.cloud.google.com/avspeech-
# files/...) requires an authenticated Google account (confirmed via `curl
# -I`: 403 / login redirect) — not actually anonymously public despite
# being "the" documented link. Verified working anonymous mirror instead
# (2026-07-26, confirmed via direct curl: 2,621,845 rows, no header, no auth
# wall): huggingface.co/datasets/bbrothers/avspeech-metadata.
AVSPEECH_TRAIN_CSV_URL = (
    "https://huggingface.co/datasets/bbrothers/avspeech-metadata/resolve/main/avspeech_train.csv"
)
AVSPEECH_DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw_avspeech"
AVSPEECH_RAW_CSV_DIR = AVSPEECH_DATA_RAW_DIR / "csv"
AVSPEECH_RAW_CLIPS_VIDEOS_DIR = AVSPEECH_DATA_RAW_DIR / "clips_videos"  # no clips_audios/ — audio is never used
AVSPEECH_PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed_avspeech"
# Scratch space for fetched subclips before cropping, deleted once cropping
# finishes — same rationale as WASD_STAGING_DIR.
AVSPEECH_STAGING_DIR = AVSPEECH_DATA_RAW_DIR / "_staging"
# Resumability manifest — one video_id per line, written the moment that
# segment is attempted (success OR failure), so a rerun with a larger
# AVSPEECH_TARGET_HOURS only fetches the NEW segments beyond what a
# previous run already tried, instead of re-fetching everything from
# scratch (added 2026-07-27 after the first full run: true real-world
# success rate was ~46.7%, far lower than an early small-sample estimate
# suggested, so most re-attempts of already-tried segments would just
# re-hit the same still-unavailable videos — not worth redoing). Lives
# directly under AVSPEECH_DATA_RAW_DIR, NOT inside AVSPEECH_STAGING_DIR,
# since staging is deleted at the end of every run.
AVSPEECH_ATTEMPTED_MANIFEST = AVSPEECH_DATA_RAW_DIR / "attempted_video_ids.txt"

# Target REALIZED hours (actual usable, post-attrition, cropped-and-labeled
# footage in {train,val}_orig.csv — see compute_realized_hours() in the
# download script), NOT raw selected-segment duration. The download script
# auto-loops — selecting progressively larger raw batches via
# select_next_batch(), sized off the empirically observed realized/raw
# success rate — until this REALIZED total is actually met, so this value
# means exactly what it says and needs no manual raw-hours translation.
# History: an earlier one-shot design targeted 67.0h as RAW hours to
# attempt, picked from an early small-sample success-rate estimate of
# ~84.2%; that run actually completed (2026-07-26) at only 31.31h realized
# — a true success rate of ~46.7%, far lower than the small sample
# suggested. That clunky "guess raw hours, manually recompute, rerun"
# workflow is exactly what this auto-looping redesign (2026-07-27)
# replaced. The original ask was 67 hours realized, unchanged here — only
# the semantics changed, not the number.
AVSPEECH_TARGET_HOURS = 67.0
# This project's own train/val split (AVSpeech ships no train/val
# partition relevant to a selected subset) — fixed-seed shuffle by
# youtube_id, same _SELECTION_SEED=42 convention as AVA/WASD's own subset
# selection.
AVSPEECH_VAL_FRACTION = 0.2

# batch-face (PyPI, MIT) — RetinaFace detection, GPU-selectable, and built
# around batched multi-frame GPU inference rather than one image at a time.
# Chosen over retinaface-pytorch (pins exact torch==1.9.0/torchvision==
# 0.10.0, would conflict with this project's installed torch) and
# retina-face (TensorFlow-based — a second, unwanted heavy ML framework +
# separate GPU/CUDA stack to reconcile). Uses PyTorch CUDA directly, so
# unlike MediaPipe's GPU delegate (see MEDIAPIPE_DELEGATE above) it's
# unaffected by this server's NVIDIA kernel/userspace driver mismatch —
# confirmed directly (2026-07-26): loaded on GPU and ran real detection
# (score 0.999 on an actual face) with no EGL-style failure.
RETINAFACE_GPU_ID = 0
# No built-in identity tracking in batch-face/RetinaFace — same face is
# picked frame-to-frame by nearest box-center distance to the previous
# frame's chosen box (first frame anchored to the CSV's (x_norm, y_norm)
# hint instead). If no detected face in a frame is within this pixel
# distance of the previous frame's chosen center, tracking is considered
# lost and the rest of that clip is skipped — never silently jump to a
# different person's face just because the real target was briefly
# undetected (occlusion, turned head, etc.).
RETINAFACE_MAX_TRACK_DISTANCE_PX = 100
# Per source-video yt-dlp invocation, mirrors NUM_WASD_DOWNLOAD_WORKERS —
# bound by YouTube-side throttling risk on concurrent connections, not
# local CPU/disk.
NUM_AVSPEECH_DOWNLOAD_WORKERS = 8

# --- scripts/modeling/train/windowed_combined.py / scripts/modeling/evaluate/windowed_combined.py
#     (added 2026-07-17) ---
# Trains WindowedSpeakingDetectorRNN from scratch on UniTalk-ASD + AVA
# combined (src/windowed_dataset.py's extra_sources=[ava_source(split)]) —
# NOT fine-tuning an existing checkpoint, a fresh run, per explicit user
# correction (see [[ava-dataset-integration]]). Own checkpoint/log/eval
# filenames, entirely separate from the UniTalk-only WINDOWED_* /
# WINDOWED_EVALUATION_DIR constants above, so this never overwrites that
# run's results — the two are meant to be compared against each other
# (does adding AVA help?), not merged.
COMBINED_WINDOWED_BEST_CHECKPOINT_FILENAME = "combined_windowed_best_model.pt"  # gitignored (checkpoints/*.pt)
COMBINED_WINDOWED_LAST_CHECKPOINT_FILENAME = "combined_windowed_last_model.pt"  # gitignored (checkpoints/*.pt)
COMBINED_WINDOWED_TRAIN_METRICS_FILENAME = "combined_windowed_train_metrics.csv"    # gitignored (logs/*.csv)
COMBINED_WINDOWED_TRAIN_LOG_FILENAME = "combined_windowed_train.log"                # gitignored (logs/*.log)
COMBINED_WINDOWED_LOSS_CURVE_FILENAME = "combined_windowed_loss_curve.png"          # gitignored (logs/*.png)
COMBINED_WINDOWED_ACCURACY_CURVE_FILENAME = "combined_windowed_accuracy_curve.png"  # gitignored (logs/*.png)
COMBINED_WINDOWED_TRAINING_REPORT_FILENAME = "combined_windowed_README.md"          # gitignored (logs/*.md)
COMBINED_WINDOWED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_windowed_combined"

# --- scripts/modeling/train/windowed_all_combined.py / scripts/modeling/evaluate/windowed_all_combined.py
#     (added 2026-07-25) ---
# Trains WindowedSpeakingDetectorRNN from scratch on UniTalk-ASD + AVA + WASD
# combined (src/windowed_dataset.py's extra_sources=[ava_source(split),
# wasd_source(split)]) — same "fresh run, not fine-tuning" rationale as
# COMBINED_WINDOWED_* above (see [[ava-dataset-integration]]), extended to a
# third data source. Own checkpoint/log/eval filenames, entirely separate
# from both WINDOWED_* (UniTalk-only) and COMBINED_WINDOWED_* (UniTalk+AVA),
# so all three runs stay directly comparable side by side (does AVA help?
# does WASD help on top of that?) instead of any one overwriting another.
ALL_WINDOWED_BEST_CHECKPOINT_FILENAME = "all_windowed_best_model.pt"  # gitignored (checkpoints/*.pt)
ALL_WINDOWED_LAST_CHECKPOINT_FILENAME = "all_windowed_last_model.pt"  # gitignored (checkpoints/*.pt)
ALL_WINDOWED_TRAIN_METRICS_FILENAME = "all_windowed_train_metrics.csv"    # gitignored (logs/*.csv)
ALL_WINDOWED_TRAIN_LOG_FILENAME = "all_windowed_train.log"                # gitignored (logs/*.log)
ALL_WINDOWED_LOSS_CURVE_FILENAME = "all_windowed_loss_curve.png"          # gitignored (logs/*.png)
ALL_WINDOWED_ACCURACY_CURVE_FILENAME = "all_windowed_accuracy_curve.png"  # gitignored (logs/*.png)
ALL_WINDOWED_TRAINING_REPORT_FILENAME = "all_windowed_README.md"          # gitignored (logs/*.md)
ALL_WINDOWED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_windowed_all_combined"

# --- scripts/modeling/train/frames_combined.py / scripts/modeling/evaluate/frames_combined.py (added 2026-07-19) ---
# Per-frame counterpart to COMBINED_WINDOWED_* above — trains SpeakingDetectorRNN
# (the per-frame many-to-many model, src/model.py) from scratch on
# UniTalk-ASD + AVA combined (src/dataset.py's extra_sources=[ava_source(split)])
# instead of UniTalk-ASD alone. Own checkpoint/log/eval filenames, entirely
# separate from the UniTalk-only BEST_CHECKPOINT_FILENAME/EVALUATION_DIR
# constants above, so this never overwrites that run's results.
COMBINED_BEST_CHECKPOINT_FILENAME = "combined_best_model.pt"  # gitignored (checkpoints/*.pt)
COMBINED_LAST_CHECKPOINT_FILENAME = "combined_last_model.pt"  # gitignored (checkpoints/*.pt)
COMBINED_TRAIN_METRICS_FILENAME = "combined_train_metrics.csv"    # gitignored (logs/*.csv)
COMBINED_TRAIN_LOG_FILENAME = "combined_train.log"                # gitignored (logs/*.log)
COMBINED_LOSS_CURVE_FILENAME = "combined_loss_curve.png"          # gitignored (logs/*.png)
COMBINED_ACCURACY_CURVE_FILENAME = "combined_accuracy_curve.png"  # gitignored (logs/*.png)
COMBINED_TRAINING_REPORT_FILENAME = "combined_README.md"          # gitignored (logs/*.md)
COMBINED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_combined"
