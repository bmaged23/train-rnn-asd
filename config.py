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

# Windowed pipeline only (src/windowed_dataset.py's
# WindowedSourceBalancedBatchSampler) — added 2026-07-29 once the combined
# dataset reached 6 sources of wildly different sizes (Columbia ~0.65% of
# train windows vs. WASD/AVSpeech ~28% each). BALANCED_BATCHES above only
# balances the SPEAKING_AUDIBLE/NOT_SPEAKING label; by itself it makes each
# source's presence per batch merely PROPORTIONAL to that source's overall
# size (empirically verified 2026-07-29 — a batch mirrors the ~19/15/28/1/28/9%
# UniTalk/AVA/WASD/Columbia/AVSpeech/SpeakingFaces split, so Columbia barely
# appears in most batches). When this flag is True instead, every batch
# still matches the dataset's overall SPEAKING/NOT_SPEAKING ratio (same
# per-class target as BALANCED_BATCHES), but WITHIN each class, every source
# that contributes to that class gets an EQUAL share of that class's slots
# in every batch, not a size-proportional one.
#
# Like every other sampler in this project, every real window is still used
# AT MOST once per epoch — NOT oversampled/undersampled. An earlier version
# (2026-07-29) instead repeated small sources (e.g. Columbia, ~30x/epoch) to
# force a perfectly equal split every single batch; rejected 2026-07-30
# (explicit user request) once it was clear that level of repetition risked
# the model memorizing Columbia's tiny pool rather than generalizing. Now,
# once a source's real supply for a class runs out, it just stops
# contributing for the rest of the epoch and its share is redistributed
# among whichever sources for that class still have data left (water-filling
# — see WindowedSourceBalancedBatchSampler._water_fill()) — composition is
# only perfectly equal-per-source in the early batches, before the smallest
# sources are exhausted; later batches lean increasingly on whichever large
# sources (WASD/AVSpeech) still have data remaining. That's the intentional
# shape, not a bug.
#
# Applied to the TRAIN split ONLY (src/windowed_dataset.py's
# get_windowed_dataloader, same train-only principle as WINDOW_TRAIN_STRIDE)
# — a bug where this also silently applied to val/test was caught and fixed
# 2026-07-30, after a training run using it produced a much worse, noisier
# result (val_loss climbing steadily instead of the usual mild overfit
# curve) than the otherwise-identical prior run without it. Root cause: the
# oversampling/undersampling the OLD version deliberately did for TRAIN was
# also distorting val during training (early-stopping/checkpoint-selection
# was deciding on a resampled, non-representative val signal, not the true
# held-out set) and test during evaluation — caught via a test-time
# diagnostic showing suspiciously identical per-source sample counts, which
# should never happen on real (undistorted) data. Whether
# BALANCED_SOURCE_BATCHES=True is actually beneficial for TRAIN is still
# untested with the current (no-repeat) implementation — every real run so
# far used either the buggy val/test leak or the oversampling version, so
# neither is a clean read on this exact behavior.
BALANCED_SOURCE_BATCHES = True

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

# A MAR_MAX_VALUE clip (1.0) was tried 2026-08-04, prompted by a real
# training run showing a sharp mid-run loss spike traced to an isolated
# misdetected frame (detected=True but physically inconsistent landmarks)
# — MAR independently spiked to 1.045 on that exact same frame (vs.
# neighboring frames' 0.25/0.036), confirming it shares
# MOUTH_GEOMETRY_MAX_DISTANCE's vulnerability, not just the new distance
# features. The clip DID stabilize the training curve (no repeat of the
# spike) but made final per-source accuracy net WORSE (Columbia dropped
# 0.575->0.507) — reverted per explicit user request after comparing real
# results, not just abandoned unverified. Worth retrying with a different
# ceiling or a different fix (e.g. filtering/repairing the specific bad
# frames instead of clipping every frame's value) if this line of work is
# revisited.

# Single pool toggle (added 2026-08-03, explicit user request) — when True,
# appends 36 raw distance features (18 height + 18 width) derived from
# MediaPipe's REAL FACE_LANDMARKS_LIPS topology (inspected directly via
# mediapipe.tasks.python.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
# not guessed). That topology is 4 arcs of 11 points each (2 shared corners
# + 9 interior positions, left-to-right): outer-lower, outer-upper,
# inner-lower, inner-upper — outer corners are 61/291, inner corners 78/308.
#
# MOUTH_HEIGHT_PAIRS (18): each lower-arc interior point paired with its
# counterpart directly above it on the matching upper arc (same position,
# left-to-right) — 9 outer + 9 inner. MAR_VERTICAL_PAIRS' 3 pairs are
# confirmed to be exactly 3 consecutive positions out of the 9 inner pairs
# here — MAR only samples 3 of these 18 and folds them into one averaged
# ratio; this exposes all 18 individually (e.g. asymmetric mouth opening,
# which MAR's averaging hides, becomes visible).
#
# MOUTH_WIDTH_PAIRS (18): same "opposite each other" principle, mirrored the
# other direction — each arc's 9 interior positions are symmetric around a
# center point, so position N-from-left-corner has a natural mirror partner
# at position N-from-right-corner ON THE SAME ARC (4 pairs/arc, the center
# position has no partner) = 16 local width pairs, plus the 2 overall
# corner-to-corner widths (outer = MAR_HORIZONTAL_PAIR itself, inner is
# new) = 18 total, deliberately matching the height count for a symmetric
# 18-height/18-width grid rather than 2 width measurements alone.
#
# Unlike MAR (a self-normalizing ratio, deliberately left UNCHANGED by
# AUGMENT_LANDMARKS' rotation/scale jitter — see that config's comment),
# these are raw distances and MUST be recomputed from the landmark
# coordinates AFTER augmentation (src/windowed_dataset.py's
# augment_landmarks()) — a scale-jittered window's mouth genuinely is wider/
# narrower in the augmented sample, and leaving a stale pre-augmentation
# value would silently feed the model a wrong number. No separate MAR-style
# ratio/division is needed for scale-invariance either way: the landmark
# COORDINATES themselves are already normalized (centered on nose tip,
# scaled by inter-ocular distance — scripts/features/extract_landmarks.py's
# _normalize_record) before anything downstream sees them, so a plain
# Euclidean distance between any two already-normalized points is itself
# already scale/position-invariant.
USE_MOUTH_GEOMETRY_FEATURES = False

# A MOUTH_GEOMETRY_MAX_DISTANCE clip (2.0) was tried 2026-08-04, prompted
# by a real training run showing a sharp, sustained loss spike starting
# mid-run (train AND val loss both got worse together, not the usual
# overfitting shape). Investigation found a heavy tail in the 36 distances
# (typical mean 0.25, 99.9th percentile 0.93, but max 3.85) from two
# distinct causes: (1) most of the tail (up to ~1.7) is genuine
# head-pose-driven signal — a raw distance is more exposed to head
# rotation/depth changes than MAR (a ratio, which partially cancels pose
# effects) — confirmed via one real case showing a smooth 10-frame
# rise-and-fall, not a discontinuity; (2) the most extreme values (2.5+)
# traced to isolated single-frame MediaPipe misdetections — a lone
# `detected=True` frame sandwiched between stretches of failed detection,
# with physically inconsistent z-coordinates between paired points
# (`detected=True` only means inference ran, not that the landmarks are
# trustworthy — a pre-existing data-quality edge case, always there but
# invisible in the raw 120 landmark-coordinate columns and dampened by
# MAR's ratio, that the new raw distances were the first feature to
# expose numerically).
#
# The clip DID stabilize the training curve (no repeat of the spike) but
# made final per-source accuracy net WORSE (Columbia dropped 0.575->0.507)
# — reverted per explicit user request after comparing real results, not
# just abandoned unverified. Worth retrying with a different ceiling or a
# different fix (e.g. filtering/repairing the specific bad frames instead
# of clipping every frame's value) if this line of work is revisited.

# (upper_landmark_index, lower_landmark_index) real MediaPipe indices — NOT
# positions within the USE_MOUTH_LANDMARKS_ONLY-filtered array, that mapping
# is derived at runtime in src/dataset.py regardless of which landmark set
# is active. 9 outer-lip pairs then 9 inner-lip pairs, left-to-right across
# the mouth; the inner set includes MAR_VERTICAL_PAIRS' 3 as a subset
# (positions 13-15 here: (82,87)==MAR pair 2, (13,14)==MAR pair 1 (center),
# (312,317)==MAR pair 3).
MOUTH_HEIGHT_PAIRS = [
    (185, 146), (40, 91), (39, 181), (37, 84), (0, 17),
    (267, 314), (269, 405), (270, 321), (409, 375),
    (191, 95), (80, 88), (81, 178), (82, 87), (13, 14),
    (312, 317), (311, 402), (310, 318), (415, 324),
]

# (left_landmark_index, right_landmark_index) real MediaPipe indices — 16
# local mirror-symmetric pairs (4 per arc: outer-lower, outer-upper,
# inner-lower, inner-upper) plus 2 overall corner-to-corner widths. First
# overall pair duplicates MAR_HORIZONTAL_PAIR (61, 291) deliberately,
# exposed as its own raw feature instead of only ever appearing inside
# MAR's ratio; the rest aren't used anywhere else in this project.
MOUTH_WIDTH_PAIRS = [
    (146, 375), (91, 321), (181, 405), (84, 314),
    (185, 409), (40, 270), (39, 269), (37, 267),
    (95, 324), (88, 318), (178, 402), (87, 317),
    (191, 415), (80, 310), (81, 311), (82, 312),
    (61, 291), (78, 308),
]

# Windowed pipeline only (src/windowed_dataset.py's WindowedLandmarkSequenceDataset)
# — cheap landmark-space augmentation, added 2026-07-30, motivated by a real
# deployment gap: every combined source is near-frontal footage (YouTube
# interviews/movies, a voice-command booth), but the actual deployment
# cameras are overhead/multi-directional. Chosen over image-space
# augmentation (warping the raw crop, then re-running MediaPipe) for two
# reasons: (1) cost — extraction is already CPU-bound (MediaPipe's GPU
# delegate is broken on this box, see MEDIAPIPE_DELEGATE below); re-running
# it on augmented copies would multiply an already-slow step, while a 3D
# rotation/scale of already-extracted (x,y,z) points is essentially free.
# (2) a flat 2D image warp isn't a physically accurate stand-in for a real
# different camera position anyway (can't reveal genuinely occluded surface
# like the underside of the chin) — landmark rotation has the same
# fundamental limitation, so this is explicitly a general pose/robustness
# regularizer, NOT a fix for the overhead-camera domain gap (that needs
# real footage from closer-to-deployment angles — see
# [[project-asd-dataset-sizes]]-style future dataset additions).
#
# Two modes, switched via AUGMENT_ON_THE_FLY, both A/B tested 2026-07-30/31
# on identical runs otherwise:
#   - On-the-fly (AUGMENT_ON_THE_FLY=True): a fresh random rotation/scale
#     applied in __getitem__ on every access — no extra stored windows, each
#     real window gets a DIFFERENT perturbation every epoch (over a 31-37
#     epoch run, ~31-37 distinct views of the same window).
#   - Materialized (AUGMENT_ON_THE_FLY=False): AUGMENT_COPIES_PER_WINDOW
#     augmented copies generated ONCE at dataset-construction time and
#     appended as extra rows alongside the originals (visible in the
#     "windows kept" count, doubles the train set at COPIES_PER_WINDOW=1) —
#     but every copy is fixed for the whole run, so the model sees the same
#     single perturbed version every epoch, and each epoch takes ~1.7-1.8x
#     longer (genuinely 2x the windows to iterate over the whole model then).
# Result: on-the-fly won on combined accuracy/F1 AND on Columbia
# specifically (F1 55.21% vs 51.25%, recall 43.69% vs 39.81% — the source
# augmentation helps most, materialized's fixed-single-copy-per-window
# limitation showed up clearest there) — see [[project-asd-windowed-sampler-and-augmentation]]
# for the full 3-way comparison table. **Decision: AUGMENT_ON_THE_FLY=True
# is the default** — better results, cheaper per epoch. Materialized is
# kept as an option (not deleted) since it's still a legitimate technique
# and the config.WINDOWED_CACHE_DIR caching layer supports both modes (the
# cache key includes the mode, so switching between them never mixes up
# cached data from the other).
#
# Both modes: applied to every source uniformly (all 6 combined datasets —
# it's inside the shared WindowedLandmarkSequenceDataset construction/
# __getitem__ logic, which doesn't distinguish sources), and to the TRAIN
# split ONLY (same principle as WINDOW_TRAIN_STRIDE/BALANCED_SOURCE_BATCHES
# — augmenting val/test would make eval signal dishonest). Materialized
# copies carry the same person_id/source_label/label as the original they
# were copied from — rotation/scale doesn't change which track a window
# came from or its majority-vote SPEAKING/NOT_SPEAKING label. MAR is left
# unchanged by the transform in both modes (a known simplification, not
# recomputed from the rotated points — MAR is a ratio, effectively
# scale-invariant, and only mildly sensitive to small rotation angles;
# revisit if AUGMENT_ROTATION_MAX_DEGREES is ever raised well beyond mild
# jitter). Zero-padded/undetected frames are unaffected either way —
# rotating or scaling the zero vector is still zero.
AUGMENT_LANDMARKS = True
AUGMENT_ON_THE_FLY = False            # True = fresh transform every epoch access (no extra stored rows); False = fixed extra copies materialized once at construction time
AUGMENT_ROTATION_MAX_DEGREES = 15.0  # independent random angle per axis (pitch/yaw/roll), uniform in [-x, x], one fresh draw per access (on-the-fly) or per copy (materialized)
AUGMENT_SCALE_RANGE = (0.9, 1.1)     # uniform random scale factor, one fresh draw per access (on-the-fly) or per copy (materialized)
AUGMENT_COPIES_PER_WINDOW = 1        # materialized mode only — how many augmented copies to append per real train window (1 = train set doubles); unused when AUGMENT_ON_THE_FLY=True

# On-disk cache for the fully-built WindowedLandmarkSequenceDataset (post
# CSV-load, post-windowing, post-augmentation) — added 2026-07-31 since
# building the combined-6-source train set from scratch (parsing tens of GB
# of CSVs, windowing, then materializing AUGMENT_COPIES_PER_WINDOW copies)
# takes several minutes every single training run. WindowedLandmarkSequenceDataset
# transparently loads a matching cache file if one exists (skipping ALL of
# that work) and writes one after a from-scratch build, so the FIRST run
# after any relevant setting changes pays the full cost once and every
# later run is near-instant. scripts/features/build_windowed_all_combined_cache.py
# explicitly (re)builds it ahead of time, matching this project's existing
# download->process->split->train pipeline-stage convention, rather than
# relying on it being silently built as a side effect of the next training
# run.
#
# TRAIN split ONLY (explicit user request, 2026-07-31) — val/test are never
# written to or read from this cache, always rebuilt from the CSVs on every
# use. Val/test are cheap to build already (tens of seconds, vs. minutes
# for train's much larger, augmented set) and are never augmented anyway,
# so caching them saves little and just means one more file that could go
# stale; only train's real cost (CSV parsing + windowing + materializing
# augmented copies) is worth avoiding paying twice.
#
# Cache validity is keyed on every setting that actually affects the built
# arrays (split, window size/stride/padding, detection-ratio thresholds,
# feature-selection toggles, augmentation settings, WINDOWED_SEED, and which
# sources are included) — see WindowedLandmarkSequenceDataset's _cache_key()
# for the exact list. A stale cache is simply never matched (different key
# -> different filename), not silently reused with wrong settings; it just
# sits there unused. Nothing deletes old cache files automatically — safe
# to clear data/windowed_cache/ by hand if disk space matters more than
# keeping old variants around.
WINDOWED_CACHE_DIR = PROJECT_ROOT / "data" / "windowed_cache"

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

# Whether to apply compute_pos_weight()'s dynamically-computed value (train
# split's own NOT_SPEAKING:SPEAKING_AUDIBLE window-label ratio) to the loss
# at all. True (default) is the existing, already-validated behavior —
# nothing changes unless this is flipped. False trains with plain unweighted
# BCE instead, e.g. to test whether pos_weight is still pulling its weight
# now that BALANCED_SOURCE_BATCHES/the 6-dataset mix has pushed the combined
# class ratio close to even (last measured pos_weight=0.8813, already only a
# mild correction — see [[project-asd-dataset-sizes]]).
APPLY_POS_WEIGHT = True

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
#
# Used by the per-frame and landmark-windowed pipelines
# (scripts/modeling/train/{frames,frames_combined,windowed,windowed_combined,
# windowed_all_combined}.py). The image CNN+LSTM pipeline
# (image_windowed_all_combined.py) has its OWN separate
# IMAGE_EARLY_STOPPING_PATIENCE instead, deliberately NOT this constant —
# see that config entry for why (2026-08-01, explicit user request: the two
# pipelines' epoch costs are different enough that one shared patience
# value doesn't fit both).
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
# best as of 2026-07-17 on the smaller dataset mix then in use) -> 3s (75
# frames, F1=0.6652 without padding, 0.6749 with WINDOW_PAD_MIN_LENGTH_RATIO=0.5
# padding — still short of 2s) -> back to 2s (used through the 6-dataset
# combined run, 2026-07-29, val_acc=92.96%/test F1=93.22%). A 5-second window
# was checked and rejected around 3s/2026-07-17: it would drop 80%+ of tracks
# as too short at the dataset mix of that time. Briefly retried at 5s with
# WINDOW_PAD_SHORT_TRACKS on (2026-07-29 night, given the 6-source mix's much
# larger track pool since — SpeakingFaces tracks alone are a fixed 804
# frames/~32s, comfortably clearing a 125-frame window several times over
# unpadded) but the user stopped that run and asked to revert to 2s/no
# padding before it produced a result — untested at this dataset mix, not
# found to be wrong, just not run to completion. Revisit later if wanted.
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
# data. Was False (no padding) through the 6-dataset combined run
# (WINDOW_SECONDS=2, 2026-07-29). Briefly re-enabled alongside WINDOW_SECONDS=5
# that same night to test whether padding recovers the "80%+ tracks too
# short" problem that got 5s rejected before, but the user stopped that run
# before it finished and asked to revert both back — set back to False here,
# matching WINDOW_SECONDS's revert to 2 above.
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
# Raised 8->24, 2026-08-27, from a REAL measured sweep against S3 (replicating
# extract_frames()' exact ffmpeg command, 30s slices from t=900, 3 reps per
# point, median). Throughput in video-seconds decoded per wall second:
#   1w 5.87 | 2w 8.09 | 4w 8.80 | 8w 11.14 | 12w 14.82 | 16w 17.04
#   16w 14.67 | 24w 18.84 | 32w 14.97 | 48w 17.49   (second run, videos reused)
# Rises monotonically to ~16 then PLATEAUS; 16-24 are equivalent within noise
# (32 scoring below both 24 and 48 is noise, not signal). Scaling is sublinear
# (16w = only 2.9x of 1w) so this is bandwidth/latency-bound, not CPU-bound.
#
# CORRECTION: an earlier edit today lowered this 8->4 on the theory that S3 was
# throttling concurrent range requests — inferred from the real run completing
# 5 videos in 30 minutes then none in the next 95. That theory was WRONG, and
# 4 is near the worst setting measured (8.80 vs 18.84). The real cause of that
# stall was almost certainly network contention: WASD and SpeakingFaces were
# both downloading hard at the same time. Run one dataset at a time.
#
# CAVEAT: the sweep used 30s slices, while real AVA videos are ~15-min
# segments. Short slices are seek/latency-dominated, which is exactly where
# concurrency helps most, so the real-world gain may be smaller than 2.9x.
NUM_AVA_DOWNLOAD_WORKERS = 24

# Per-video ffmpeg wall-clock ceiling in scripts/dataset/download_ava_subset.py's
# extract_frames(). Was a hardcoded 1800 (30 min); a REAL run on 2026-08-27 died
# on lWXhqIAvarw.mkv, which needs 865.6 video-seconds decoded and blew past it —
# and because subprocess.run() RAISES TimeoutExpired rather than returning, that
# single video aborted the whole 153-video job at 45/120 complete.
#
# Raised to 5400 (90 min) for headroom on the long tail. Note the interaction
# with NUM_AVA_DOWNLOAD_WORKERS: more concurrent workers means less bandwidth
# each, so each video decodes SLOWER and is MORE likely to hit this ceiling.
# Raising worker count without raising this trades throughput for fragility.
AVA_FFMPEG_TIMEOUT_SECONDS = 5400

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
# Lowered 8->3, 2026-08-30. At 8 workers with NO request pacing, a real run
# fired ~16k yt-dlp calls and YouTube rate-limited then bot-blocked this IP
# ~45 MINUTES in (verified from crop timestamps: ~500 clips/15min before
# 08:30, <9/15min for the following 6 hours). WASD uses 8 safely because it
# only has 164 source videos; AVSpeech has 16k+ — two orders of magnitude
# more requests, so the same setting is not remotely equivalent.
# ADAPTIVE CONCURRENCY (2026-08-31). Rather than one fixed worker count, the
# script now runs a governor that moves between FLOOR and CEILING at runtime:
#   - on a YouTube block: HALVE concurrency (never below FLOOR), pause every
#     worker for BLOCK_COOLDOWN_SECONDS, then carry on. The blocked source's
#     videos are simply NOT marked attempted, so they come back round later
#     and the candidate pool is never poisoned.
#   - after HEALTHY_STREAK consecutive good source videos: +1 worker, up to
#     CEILING. So a clean link drifts back up to full speed on its own.
# This is the classic additive-increase / multiplicative-decrease shape: back
# off hard on trouble, recover gently. NUM_AVSPEECH_DOWNLOAD_WORKERS is the
# STARTING point only; the governor owns it after that.
AVSPEECH_WORKERS_FLOOR = 3        # measured safe: 22h unblocked at this level
AVSPEECH_WORKERS_CEILING = 8      # ~2x the floor's throughput when the link allows
AVSPEECH_BLOCK_COOLDOWN_SECONDS = 1800   # BASE cooldown; doubles per consecutive
                                          # block (see AVSPEECH_BLOCK_COOLDOWN_MAX).
                                          # A fixed 30min caused a 13h thrash on
                                          # 2026-09-01: wake, get blocked, wait 30min,
                                          # repeat — 35 blocks, zero recoveries.
AVSPEECH_BLOCK_COOLDOWN_MAX_SECONDS = 14400   # 4h ceiling on the exponential backoff
# Throughput watchdog. The governor originally detected trouble ONLY by matching
# yt-dlp error strings, so a failure mode with no matching string was invisible:
# on 2026-09-01 a missing JS challenge solver made every download fail with
# "Requested format is not available", which matched nothing, so every failed
# video was counted as healthy and the run produced nothing for 5.5 hours.
# This is the backstop: if a window of attempts yields (almost) no clips, treat
# it as a block regardless of what the errors said.
AVSPEECH_WATCHDOG_WINDOW = 40        # source videos per evaluation window
AVSPEECH_WATCHDOG_MIN_PRODUCTIVE = 2 # fewer productive videos than this in a window -> block   # 30 min; the real block on 2026-08-30 cleared in ~7h,
                                          # but a short pause is usually enough for soft 429s
AVSPEECH_HEALTHY_STREAK_TO_SCALE_UP = 150  # good source videos between step-ups
# Governor STARTING point only (it owns the value after that, moving between
# FLOOR and CEILING). Set to 4 on 2026-09-01: starting at the 8 ceiling is what
# triggered the block that cost ~13h, and additive-increase now works, so the
# governor will climb to 8 on its own if the link demonstrably tolerates it.
NUM_AVSPEECH_DOWNLOAD_WORKERS = 3

# Request pacing passed straight through to yt-dlp. Without these the script
# has no throttle of any kind — the direct cause of the block above.
# --sleep-requests spaces metadata requests; --sleep-interval/--max-sleep-interval
# add a random pause before each download (randomised so the traffic doesn't
# look metronomic).
AVSPEECH_YTDLP_SLEEP_REQUESTS = 2.0
AVSPEECH_YTDLP_SLEEP_INTERVAL = 1.0
AVSPEECH_YTDLP_MAX_SLEEP_INTERVAL = 5.0

# Substrings that mean "YouTube is refusing us", not "this video is gone".
# Hitting one aborts the whole run: continuing would keep writing perfectly
# good video ids into the attempted-manifest as though they had been fairly
# tried, permanently removing them from the candidate pool. That is exactly
# what cost 58,064 ids on 2026-08-30 and had to be repaired by hand.
AVSPEECH_BLOCK_SIGNATURES = (
    "Sign in to confirm you're not a bot",
    "Sign in to confirm you\u2019re not a bot",
    "HTTP Error 429",
    "Too Many Requests",
    "confirm you are not a bot",
)

# --- scripts/dataset/download_speakingfaces_subset.py (added 2026-07-28) ---
# SpeakingFaces (ISSAI, "SpeakingFaces: A Large-Scale Multimodal Dataset of
# Voice Commands with Visual and Thermal Video Streams", CC-BY-4.0) — a
# sixth raw source, added to grow the NOT_SPEAKING class instead (the
# opposite gap from AVSpeech): 142 lab-recorded subjects, each doing 2
# trials of 2 sessions (Session 1 silent-and-still, Session 2 reading voice
# commands) across 9 fixed camera angles. Only Session 1 (silent) is used —
# see download script's module docstring for why Session 2 is skipped
# entirely (its public release only ships ~4-6 sparse, non-consecutive
# frames per spoken command, not continuous video, so it can't support this
# project's 50-frame windowed model at all; confirmed via a pilot download
# and frame-index gap-check 2026-07-28 before committing to the source).
# Session 1 IS fully continuous — verified 900/900 frames present with zero
# gaps for every camera position checked — so every frame gets label_id=0
# (NOT_SPEAKING) unconditionally, mirroring AVSpeech's constant-label
# approach but for the opposite class.
#
# Hosted directly on HuggingFace as one zip per subject
# (issai/Speaking_Faces, image_only/sub_{N}_io.zip) — no YouTube re-fetch,
# no throttling risk, unlike WASD/AVSpeech. sub_2_io.zip is a known-corrupt
# 0-byte file on the source repo itself (confirmed via the HF API's own
# file-size metadata) — skipped automatically, not a bug in this script.
#
# Ships PNGs, not JPGs — kept as-is (config.IMAGE_EXTENSIONS already
# includes ".png", and process_dataset.py's DatasetProcessor only ever
# reads image_name.stem, never assumes .jpg) rather than re-encoding, to
# avoid a wasted lossy recompression pass over ~2.3M images.
#
# Each subject's zip contains 3 image "modalities" per position (thermal,
# raw rgb, and aligned rgb — the visual-thermal-synchronized one, filename's
# last field == "3") plus duplicate unaligned copies; only the aligned rgb
# stream is extracted. person_id = f"sub{{N}}_trial{{T}}_pos{{P}}" — a
# globally unique key (not the AVSpeech-style shared-constant mistake) even
# though the on-disk layout nests position under a
# f"sub{{N}}_trial{{T}}"-named video_id folder.
SPEAKINGFACES_HF_REPO_ID = "issai/Speaking_Faces"
SPEAKINGFACES_NUM_SUBJECTS = 142
SPEAKINGFACES_NUM_TRIALS = 2
SPEAKINGFACES_NUM_POSITIONS = 9
SPEAKINGFACES_FRAMES_PER_POSITION = 900  # verified continuous 1..900, no gaps
# Native capture rate, 28fps — UNLIKE download_columbia_subset.py's
# native-fps exception, this source IS resampled down to config.DATASET_FPS
# (25) so a "50-frame window" means the same 2 real seconds across every
# combined source (at native 28fps, 50 raw frames would only be ~1.79s —
# would silently break windowed-model comparability across datasets).
# Resampled via nearest-frame selection onto a 25fps grid (900 frames over
# 32.14s at 28fps -> ~804 frames at 25fps), not ffmpeg, since the source is
# already-extracted discrete PNG frames, not a video file.
SPEAKINGFACES_NATIVE_FPS = 28
SPEAKINGFACES_ALIGNED_RGB_MODALITY = "3"  # filename's last field for the visual-thermal-aligned rgb stream
SPEAKINGFACES_DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw_speakingfaces"
SPEAKINGFACES_RAW_CSV_DIR = SPEAKINGFACES_DATA_RAW_DIR / "csv"
SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR = SPEAKINGFACES_DATA_RAW_DIR / "clips_videos"

# scripts/dataset/crop_speakingfaces_faces.py output (added 2026-07-31) — a
# face-only-cropped MIRROR of SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR, for the
# candidate raw-pixel CNN+LSTM approach (see [[project-asd-windowed-sampler-and-augmentation]]
# for the landmark-based approaches this is an alternative to). The
# landmark pipeline above never needed this — MediaPipe's FaceLandmarker
# does its own internal face detection regardless of input framing, and
# extracted landmarks get normalized (centered/scaled) immediately after,
# which absorbs away any framing difference. A raw-pixel CNN gets no such
# normalization for free, and SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR's frames
# are full uncropped scenes (464x348, head+shoulders+background) unlike
# every other source's tight face-only crops — confirmed directly
# (2026-07-31) by comparing sample crop sizes/visuals across all 6 sources.
# NEVER writes into or modifies SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR itself —
# a new, separate, derived directory, same non-destructive principle as
# [[feedback-asd-no-redownload]] (original raw material is never mutated in
# place, even for a fix, only ever added to).
SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR = SPEAKINGFACES_DATA_RAW_DIR / "clips_videos_face_cropped"
# Extra margin added around each detected RetinaFace box before cropping —
# a zero-margin tight box tends to clip the chin/forehead/mouth edges,
# matching the visible margin observed in this project's other tight-crop
# sources (WASD/AVSpeech samples, eyeballed 2026-07-31) rather than an
# artificially exact face-only box no other source actually uses.
SPEAKINGFACES_FACE_CROP_MARGIN = 0.3
# Settled on plain sequential GPU after three concurrency attempts, all
# 2026-07-31: (1) one shared GPU RetinaFace detector across 8 threads, (2)
# one separate GPU detector per thread at 4 workers — both hit the exact
# same reproducible `RuntimeError: NVML_SUCCESS ==
# DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED` inside PyTorch's
# CUDA caching allocator (this box's known NVML/driver mismatch, see
# [[project-asd-wasd-download]], surfacing under concurrent CUDA allocator
# pressure — not a model-sharing artifact, confirmed by both failing
# identically). (3) CPU instead (gpu_id=-1, sidesteps CUDA/NVML entirely) —
# correctness was fine but throughput wasn't competitive: single-track CPU
# detection plateaus around 14.6s/track even at its best measured thread
# count (32 intra-op threads), already slower than GPU's 9s/track, and
# multi-worker CPU concurrency showed additional unexplained slowdown
# beyond that rather than the hoped-for multiplication. Decision: GPU,
# gpu_id=0, sequential — the only configuration that's actually been
# reliable AND fast (~9s/track, 100% success on angled + frontal
# positions, 2,538 tracks ~6.3h total). Revisit concurrency once the
# underlying NVML/driver mismatch is actually fixed (needs the box admin,
# still unresolved as of [[project-asd-wasd-download]]'s last check).
SPEAKINGFACES_CROP_GPU_ID = 0
NUM_SPEAKINGFACES_CROP_WORKERS = 1

SPEAKINGFACES_PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed_speakingfaces"
# Scratch space for each subject's downloaded zip, deleted right after that
# subject's frames are extracted — same rationale as AVSPEECH_STAGING_DIR,
# avoids keeping the ~1.2TB of raw zips AND the extracted frames on disk at
# once.
SPEAKINGFACES_STAGING_DIR = SPEAKINGFACES_DATA_RAW_DIR / "_staging"
# Resumability manifest — one "sub{N}" per line, written only once that
# subject's frames are fully extracted (see [[avspeech-manifest-bug]]-style
# caution: append-only, never rewritten in place).
SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST = SPEAKINGFACES_DATA_RAW_DIR / "completed_subjects.txt"
# This project's own train/"val" (pre-val/test-split) partition uses the
# dataset's own official per-subject Train/Validation/Test column
# (metadata/subjects.csv) rather than inventing a new fixed-seed shuffle —
# Validation+Test subjects both fold into this project's raw "val" pool,
# which scripts/splits/split_speakingfaces_val_test.py then re-splits into
# val_split/test_split the same stratified way as every other source.
NUM_SPEAKINGFACES_DOWNLOAD_WORKERS = 4  # HF Hub downloads over plain HTTPS, not YouTube — a few concurrent zip fetches is fine

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
# scripts/modeling/evaluate/windowed_all_combined_per_source.py's own output
# dir — kept separate from ALL_WINDOWED_EVALUATION_DIR above since it writes
# one metrics file per source (unitalk/ava/wasd/columbia/avspeech/
# speakingfaces/combined) rather than one combined-only report.
ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR = PROJECT_ROOT / "evaluation_windowed_all_combined_per_source"

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

# --- scripts/modeling/train/image_windowed_all_combined.py / scripts/modeling/evaluate/image_windowed_all_combined.py
#     (added 2026-07-31) ---
# Third candidate architecture, alongside the per-frame and windowed
# landmark-RNN models above: a CNN+LSTM consuming face-crop IMAGES directly
# (src/image_dataset.py), no MediaPipe landmark extraction step at all. NOT
# proposed as a fix for the overhead/multi-directional-camera deployment
# gap discussed alongside AUGMENT_LANDMARKS above (architecture-independent
# data problem) — the motivation is capturing visual cues landmarks
# discard entirely (texture, teeth/tongue visibility, motion blur), tried
# as a genuinely different model for comparison. Same 6 combined sources,
# same windowing/majority-vote-label concept as the landmark pipeline, same
# TRAIN/val/test separation via LOGICAL_SPLITS — only the per-frame
# representation changes (an image tensor instead of a landmark vector).
#
# Reuses each source's own RAW_CLIPS_VIDEOS_DIR (the face-crop JPGs the
# landmark pipeline already reads from) directly — EXCEPT SpeakingFaces,
# which uses SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR instead of
# SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR (its raw frames are uncropped full
# scenes, unlike every other source — see that constant's comment). Labels/
# track structure are read from the SAME combined_landmarks.csv files the
# landmark pipeline already produces (just the person_id/video_id/
# image_name/detected/target_index columns, not the landmark float
# columns — much cheaper to load) rather than a separate label file, so
# both pipelines stay in sync automatically if a dataset is ever
# re-extracted.
IMAGE_SIZE = 96  # square resize target — kept modest given the x-window_size compute multiplier (224x224 ImageNet-standard would be much heavier)
# ImageNet mean/std (standard torchvision convention) since IMAGE_CNN_BACKBONE
# below is ImageNet-pretrained — using its own expected input statistics.
IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_STD = (0.229, 0.224, 0.225)
# Non-square crops are padded to square (not stretched) before resizing —
# a direct resize would distort face proportions differently depending on
# each source's own native crop aspect ratio.
IMAGE_PAD_TO_SQUARE = True

# Minimum real (pre-resize) width/height, in pixels, for a face-crop image to
# be used at all — checked against the crop's SMALLER dimension, before the
# pad-to-square/resize step. A crop this tiny is mostly upsampling artifact
# once blown up ~6x to IMAGE_SIZE=96 (blurry, information-poor) rather than
# real signal, so it's treated the same as a missing file/padding frame: an
# all-zero IMAGE_SIZE x IMAGE_SIZE tensor (src/image_dataset.py's
# ImageWindowedDataset._load_image), not silently upsampled and fed to the
# model as if it were a real, informative frame. Added 2026-08-01 after a
# real-data check showed crop sizes vary hugely by source — UniTalk-ASD in
# particular has a long tail of very small crops (13.8% of a random 500-frame
# sample were under 16px on the smaller side, vs. ~0% for AVA/WASD/Columbia/
# AVSpeech/SpeakingFaces). 16 was chosen as a conservative floor — catches
# clearly-degenerate crops without discarding a large fraction of UniTalk's
# usable data (32px would have zeroed 41.6% of its frames, not just the
# genuinely-tiny ones). 0 disables the check entirely (every crop used
# regardless of size — the original, pre-2026-08-01 behavior).
IMAGE_MIN_CROP_SIZE = 16

# Maximum fraction of a window's WINDOW_SIZE frames allowed to be "masked"
# (zero-tensor substituted — a missing file, an IMAGE_MIN_CROP_SIZE-filtered
# too-small crop, or short-track padding) before the whole window is DROPPED
# from the dataset entirely, rather than kept as a sample the LSTM sees as
# if it had real temporal signal in every position. Checked once at
# dataset-construction time (src/image_dataset.py's
# ImageWindowedDataset.__init__) — a window either is or isn't included in
# the fixed sample count every batch sampler relies on, so this can't be
# decided lazily per-epoch the way IMAGE_MIN_CROP_SIZE's zero-masking is.
# 1.0 disables the check entirely (every window kept regardless of
# masked-frame count — the behavior before this was added). Added
# 2026-08-01 alongside IMAGE_MIN_CROP_SIZE, same motivation: a window that's
# mostly zero-tensors shouldn't silently look like a real training/eval
# example.
IMAGE_MAX_MASKED_FRAME_RATIO = 0.5

# Where src/image_dataset.py's build_image_cache() writes its pre-resized
# frame cache — added 2026-08-01 after profiling showed the image pipeline's
# real bottleneck wasn't the GPU (one training step's forward+backward+
# optimizer.step measured at ~43ms) but per-epoch CPU decode+resize: every
# _load_image call used to decode the ORIGINAL source crop (some sources up
# to ~970px) and pad-to-square+resize it to IMAGE_SIZE from scratch, EVERY
# epoch, for every frame in every window it appears in — real epoch times
# (1700-2100s) vs. the tiny per-step GPU cost confirmed this dominates wall
# time. build_image_cache() now does that decode+pad+resize (and the
# IMAGE_MIN_CROP_SIZE filtering) ONCE per run, up front, writing small
# already-96x96 JPEGs here; ImageWindowedDataset/_load_image then only ever
# read from this directory, not the original raw source images, so a
# missing/too-small frame is simply never written here (indistinguishable
# from "missing" downstream, same zero-mask fallback as before).
#
# Persisted and REUSED across runs when still valid — same principle as
# WINDOWED_CACHE_DIR. Initially (2026-08-01) built fresh on every single
# training run per explicit user request; reversed the same day once a
# from-scratch build of the full 6-source/3-split combined set was timed at
# ~25 min (21.2M frames total: 16.0M train/3.7M val/1.6M test) — re-paying
# that on every run (e.g. train immediately followed by eval, or two
# training runs back to back) was pure waste whenever nothing that affects
# the cache's actual content had changed. Validity keyed on IMAGE_SIZE +
# IMAGE_MIN_CROP_SIZE only (the two settings that determine what a cached
# file contains) via src/image_dataset.py's _cache_meta.json — a settings
# change invalidates and fully rebuilds the whole cache (no partial/mixed
# reuse across different settings); an unchanged split is skipped entirely,
# not rebuilt. Gitignored, scratch-only (measured real size: 81GB for the
# full 6-source/3-split set at IMAGE_SIZE=96).
IMAGE_RESIZED_CACHE_DIR = PROJECT_ROOT / "data" / "image_resized_cache"

# Worker count for ImageWindowedDataset.__init__'s construction-time
# cache-existence check (one Path.exists() syscall per real frame) —
# parallelized via ThreadPoolExecutor, which is the right tool here: a bare
# syscall releases the GIL for its whole duration, so threads give real
# concurrency with none of ProcessPoolExecutor's per-worker startup/pickling
# cost. NOT used for build_image_cache()'s actual decode/resize/encode work
# — see IMAGE_CACHE_BUILD_WORKERS below for why that one needs processes,
# not threads. Kept at 32 (not higher) since this box is shared with other
# users' GPU/CPU workloads — leaves headroom, same reasoning as
# crop_speakingfaces_faces.py's NUM_SPEAKINGFACES_CROP_WORKERS.
IMAGE_FRAME_CHECK_WORKERS = 32

# Worker count for src/image_dataset.py's build_image_cache() — decode +
# pad-to-square + resize + re-encode + write, per real frame (the actual
# preprocessing work, now done once per run instead of once per epoch).
# Uses ProcessPoolExecutor, NOT ThreadPoolExecutor — measured directly
# 2026-08-01: threading this made it SLOWER than single-threaded (2370
# frames/sec at 1 thread vs. ~500-550 frames/sec at 8-64 threads), meaning
# Pillow's resize/save calls don't release the GIL well enough here for
# threads to help; ProcessPoolExecutor (real, separate-GIL parallelism)
# measured 9,100-10,400 frames/sec across 8-32 processes on a realistic
# 40,000-frame sample, peaking at 16 (~10,400 frames/sec, ~26 min
# extrapolated for the full 16M-frame train split) — higher counts (24, 32)
# measured WORSE due to process-spawn/import overhead (every worker
# re-imports image_dataset.py's full dependency chain, including torch/
# torchvision) outweighing the extra parallelism on this box. 16 chosen as
# the measured sweet spot, not a guess.
IMAGE_CACHE_BUILD_WORKERS = 16

# torchvision.models.mobilenet_v3_small, ImageNet-pretrained — chosen for
# being lightweight (fast per-frame forward pass, since every WINDOW_SIZE
# window needs one CNN forward pass PER FRAME) over a heavier backbone
# (ResNet50/EfficientNet) that would multiply the already-heavy
# window_size x batch_size cost further. Was frozen by default
# (IMAGE_CNN_FREEZE_BACKBONE) — this project's combined dataset, while
# large by its own standards (235.51h, see [[project-asd-dataset-sizes]]),
# is still far smaller than ImageNet-scale pretraining data, so fine-tuning
# the whole backbone risked overfitting faster than it helped; only the
# LSTM+head trained from scratch initially. Unfrozen 2026-08-22 (explicit
# user request) after the frozen-backbone baseline (2026-08-02) lost badly
# to the landmark model on every hard source — watch for overfitting more
# closely than the frozen baseline needed to.
IMAGE_CNN_BACKBONE = "mobilenet_v3_small"
IMAGE_CNN_EMBED_DIM = 576  # mobilenet_v3_small's final feature-map channel count, after global average pooling
IMAGE_CNN_FREEZE_BACKBONE = False

# Dual-backbone encoder (added 2026-08-24, explicit user request): AdaFace
# IResNet100, trained on WebFace4M via the official mk-minchul/AdaFace repo
# (MIT license, verified — no commercial restriction), runs alongside
# IMAGE_CNN_BACKBONE above instead of replacing it. Per-frame embeddings
# from both are concatenated before the LSTM — see src/image_model.py and
# src/adaface_iresnet.py. Rationale: AdaFace's margin-based face-verification
# training objective pushes toward IDENTITY-invariant features (same person
# should embed similarly whether talking or silent) — a real risk that
# freezing the whole thing hands the LSTM embeddings with speaking-relevant
# motion trained OUT of them. MobileNetV3 (already fully trainable, see
# IMAGE_CNN_FREEZE_BACKBONE=False above) has no such bias but also no strong
# pretrained face-region prior of its own — concatenating hedges between
# the two rather than picking one.
IMAGE_USE_ADAFACE_BRANCH = True
IMAGE_ADAFACE_CHECKPOINT_PATH = MODELS_DIR / "adaface_ir101_webface4m.ckpt"  # gdrive link in AdaFace repo's README
IMAGE_ADAFACE_NUM_LAYERS = 100  # IResNet100 — verified against the checkpoint's own state_dict (49 blocks: 3+13+30+3)
IMAGE_ADAFACE_INPUT_SIZE = 112  # AdaFace's own native resolution — see adaface_preprocess()'s resize step
IMAGE_ADAFACE_EMBED_DIM = 512  # AdaFace's own output_layer Linear(25088, 512) — fixed by the checkpoint, not tunable
# Partial freeze, not full: Stages 1-3 (3+13+30=46 blocks) stay frozen —
# generic, task-agnostic face-region detectors worth protecting, and this
# project's data (a few hundred thousand frames) is tiny next to the 4M+
# identities AdaFace was pretrained on, so fully unfreezing risks
# catastrophically forgetting them. Stage 4 (the final 3 blocks, closest to
# the embedding, most task-specific) + the embedding head (output_layer)
# stay trainable — the natural place to let fine-tuning adapt away from
# AdaFace's identity-invariance bias toward speaking-relevant motion.
# N=3 chosen 2026-08-24 (explicit user request) as the conservative default
# — see project memory / this date's conversation for the alternative
# (Stage 3+4, 33 blocks) that was considered and not chosen.
IMAGE_ADAFACE_PARTIAL_UNFREEZE_NUM_BLOCKS = 3
IMAGE_COMBINED_EMBED_DIM = IMAGE_CNN_EMBED_DIM + (IMAGE_ADAFACE_EMBED_DIM if IMAGE_USE_ADAFACE_BRANCH else 0)

# Same architecture shape as WINDOWED_HIDDEN_SIZE/WINDOWED_NUM_RNN_LAYERS
# above (bidirectional LSTM over the window, final hidden state -> linear
# head) — starting from the same capacity as a reasonable baseline, not
# independently tuned yet.
IMAGE_HIDDEN_SIZE = 256
IMAGE_NUM_RNN_LAYERS = 2

# Lowered 64->16, 2026-08-24 (explicit user request), after adding the
# AdaFace dual-backbone branch (see IMAGE_USE_ADAFACE_BRANCH) made VRAM a
# real constraint again — the old reasoning below (raised to 64 because
# VRAM wasn't the bottleneck) no longer holds now that a 65M-param, 49-block
# backbone runs on every frame. Real steady-state benchmarks on this box
# (device_map-free, single GPU): batch_size=64 OOMs outright; 32 fits but
# only leaves ~10GB free; 16 fits with ~25GB free AT THE SAME real per-epoch
# training time as every size from 8-32 (throughput is flat across that
# whole range for this architecture — no compute-efficiency loss from
# choosing the safer, smaller batch). ~3.2h/epoch on the full 6-source
# combined train set at this size.
IMAGE_BATCH_SIZE = 16

# Eval/inference-only batch size (added 2026-08-25, explicit user request)
# — separate from IMAGE_BATCH_SIZE because evaluation has no backward pass,
# so none of the gradient/activation-for-backward memory that capped
# training at 16 applies here. Real benchmarked (AMP+TF32, this box's
# current contention level): 32 leaves a healthy ~15GB free; 64 fits but
# only ~7GB free (risky given this box's observed contention spikes); 96
# OOMs outright. Chose 32 for real margin rather than the tightest number
# that happens to fit — eval only runs once per checkpoint, not worth
# cutting it close.
IMAGE_EVAL_BATCH_SIZE = 32

# Unlike the landmark pipeline's dataloaders (num_workers=0 default — an
# in-RAM array lookup per __getitem__ is cheap enough not to need it), each
# image __getitem__ call decodes WINDOW_SIZE real (pre-resized, see
# IMAGE_RESIZED_CACHE_DIR) JPG files from disk — genuinely worth
# parallelizing across worker processes.
#
# Lowered 64 -> 32 (2026-08-01) after 64 caused a REAL training run to die
# silently (consistent with an OOM-kill) with the system's 8GB swap fully
# exhausted and throughput swinging 4x (15.7 -> 3.98 it/s) from swap
# thrashing before that. Root cause: each DataLoader worker is a forked
# process, and CPython's reference counting touches (and copy-on-write
# duplicates) a page merely by READING an object on it — with this
# dataset's ~488K-window train split holding ~24M path strings in
# self.window_paths, enough of that gets touched per-worker that 64 workers
# each ballooned to ~24GB RSS, no longer sharing those pages. This is a
# different failure mode than the earlier 8->64 MediaPipe-extraction-worker
# tuning this constant's value used to cite ([[project-asd-wasd-download]])
# — that workload didn't fork a comparably large shared Python object, so
# it never hit this. Measured directly this time (3-min smoke test per
# worker count, full real train dataset, throughput + system-wide free/swap
# sampled every 10s): 16 workers -> 7.03 it/s, memory plateaus immediately
# at ~84GB system-used, zero swap growth over the whole trial. 32 workers ->
# 7.61 it/s (~8% faster), plateaus at ~101GB system-used, also zero swap
# growth. 32 chosen as the measured sweet spot — meaningfully faster than
# 16, still far from whatever count would repeat the 64-worker failure
# (not bisected further; revisit if this box's other memory pressure changes).
IMAGE_NUM_DATALOADER_WORKERS = 32

# val/test are ~50x smaller than train (9,740 vs 495,379 windows for the
# 6-source combined set) — don't need anywhere near IMAGE_NUM_DATALOADER_WORKERS
# workers of their own. Matters more than it looks: with
# persistent_workers=True (see get_image_windowed_dataloader), a script that
# builds a train_loader AND a val_loader up front keeps BOTH worker pools
# alive simultaneously for the whole run, not just whichever one is actively
# iterating — using IMAGE_NUM_DATALOADER_WORKERS for both meant 128 resident
# worker processes at once, which pushed this box into swapping (measured
# 2026-07-31: 0MiB swap free, epoch time got WORSE after adding
# persistent_workers, not better, until this was split out).
IMAGE_VAL_NUM_DATALOADER_WORKERS = 8

IMAGE_WINDOWED_SEED = 42

# Own early-stopping patience, separate from EARLY_STOPPING_PATIENCE (used
# by every other training script) — 2026-08-01, explicit user request.
# Lowered to 3 specifically for this pipeline: even after the data-loading
# fixes (build_image_cache() reuse, batched/chunked existence checks,
# torchvision.io fast decode path), each epoch here still costs ~20min —
# EARLY_STOPPING_PATIENCE=20's original justification (checkpoint selection
# already protects against post-best overfitting, so a high patience only
# wastes compute) still holds in principle, but at this pipeline's
# per-epoch cost a patience of 20 means up to ~6.5 wasted HOURS past the
# real best epoch, not a few wasted minutes like the much faster landmark
# pipeline runs that motivated 20 in the first place.
IMAGE_EARLY_STOPPING_PATIENCE = 3

# Image-space augmentation — same core principle as AUGMENT_LANDMARKS (see
# [[project-asd-windowed-sampler-and-augmentation]]): ONE random draw per
# window, applied identically to every frame in that window (one simulated
# camera/lighting condition per continuous capture, not per-frame),
# train-split only (config.AUGMENT_IMAGES and split == "train" — see
# src/image_dataset.py). Richer than the landmark version though —
# rotation/scale/brightness/contrast, real pixel-space effects a landmark
# point cloud can't represent.
#
# Same dual-mode switch as AUGMENT_ON_THE_FLY too, chosen by
# IMAGE_AUGMENT_ON_THE_FLY below: True (default) draws a fresh random
# transform every __getitem__ access (every epoch sees a different view of
# each window); False MATERIALIZES IMAGE_AUGMENT_COPIES_PER_WINDOW extra
# window entries per real window at construction time, each with ONE frozen
# draw reused every epoch — real windows and their extra materialized
# copies exist side by side, train set size grows by
# (1 + IMAGE_AUGMENT_COPIES_PER_WINDOW)x, same shape as the landmark
# pipeline's materialized mode.
#
# Deliberately NOT a literal on-disk pixel cache, unlike WINDOWED_CACHE_DIR
# for the landmark pipeline: materializing here only means storing an extra
# (window_paths, frozen_augment_params) entry per copy — a few floats, not
# decoded pixels — since ImageWindowedDataset never holds decoded pixel
# arrays in memory in the first place (images are lazily decoded from disk
# in __getitem__ either way, real or materialized). An actual pixel cache
# would be enormous (a single 6-source train pass alone is
# window_size x IMAGE_SIZE^2 x 3 bytes x ~68k windows ≈ 94GB, uncompressed,
# BEFORE any augmented copies) and WINDOWED_CACHE_DIR's own motivation (a
# multi-minute-per-run CSV-load/windowing build step worth avoiding on
# reruns) doesn't apply here either — image_dataset.py's construction step
# is cheap path/CSV bookkeeping, not the expensive part.
AUGMENT_IMAGES = False
IMAGE_AUGMENT_ON_THE_FLY = False  # True: fresh draw per access (default, matches AUGMENT_ON_THE_FLY's winning result). False: materialized frozen copies.
IMAGE_AUGMENT_COPIES_PER_WINDOW = 1  # only used when IMAGE_AUGMENT_ON_THE_FLY is False

# Materialized copies only (IMAGE_AUGMENT_ON_THE_FLY=False) — how far, in
# seconds, a copy's SOURCE FRAMES are shifted from its real window's own
# position within the same track, so the copy is genuinely different
# (continuous, real) footage, not just a pixel-transform of frames already
# covered by an existing real window. Real windows already tile each track
# every WINDOW_TRAIN_STRIDE frames (50% overlap) — a shift that happens to
# be a whole multiple of that stride would land exactly on another
# already-included real window's start, reproducing its exact frame range
# verbatim under a different name (caught 2026-07-31 before this shipped:
# a plain 2.0s shift IS exactly 2 stride-steps, since WINDOW_TRAIN_STRIDE is
# half of WINDOW_SIZE). src/image_dataset.py's _choose_shifted_start()
# handles this: internally adds half a stride on top of whatever this is
# set to, so the copy's start always falls in the GAP between two existing
# real windows instead of on one, regardless of the value here — then tries
# shifting forward first, backward second, and only reuses the original
# window's own frames (pixel-transform-only, no content shift) as a last
# resort, for tracks too short to shift within at all (near WINDOW_SECONDS
# long — most UniTalk-ASD tracks are only ~2.2s median, see
# [[project-asd-dataset-sizes]]; longer sources have far more room). Every
# candidate window (original or shifted) is always one CONTINUOUS run of
# frames, never a scattered/non-contiguous sample.
IMAGE_AUGMENT_SHIFT_SECONDS = 2.0
IMAGE_AUGMENT_ROTATION_MAX_DEGREES = 15.0  # same value as AUGMENT_ROTATION_MAX_DEGREES, for rough parity with the landmark pipeline's setting
IMAGE_AUGMENT_SCALE_RANGE = (0.9, 1.1)     # same value as AUGMENT_SCALE_RANGE
IMAGE_AUGMENT_BRIGHTNESS = 0.2             # PIL ImageEnhance.Brightness factor drawn from [1-this, 1+this]
IMAGE_AUGMENT_CONTRAST = 0.2               # PIL ImageEnhance.Contrast factor drawn from [1-this, 1+this]

# Own checkpoint/log/eval filenames, entirely separate from every landmark
# variant's files above, so this never overwrites (or is overwritten by)
# any of them — all meant to sit side by side for comparison.
IMAGE_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME = "image_all_windowed_best_model.pt"  # gitignored (checkpoints/*.pt)
IMAGE_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME = "image_all_windowed_last_model.pt"  # gitignored (checkpoints/*.pt)
IMAGE_ALL_WINDOWED_TRAIN_METRICS_FILENAME = "image_all_windowed_train_metrics.csv"    # gitignored (logs/*.csv)
IMAGE_ALL_WINDOWED_TRAIN_LOG_FILENAME = "image_all_windowed_train.log"                # gitignored (logs/*.log)
IMAGE_ALL_WINDOWED_LOSS_CURVE_FILENAME = "image_all_windowed_loss_curve.png"          # gitignored (logs/*.png)
IMAGE_ALL_WINDOWED_ACCURACY_CURVE_FILENAME = "image_all_windowed_accuracy_curve.png"  # gitignored (logs/*.png)
IMAGE_ALL_WINDOWED_TRAINING_REPORT_FILENAME = "image_all_windowed_README.md"          # gitignored (logs/*.md)
IMAGE_ALL_WINDOWED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_image_windowed_all_combined"
IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR = PROJECT_ROOT / "evaluation_image_windowed_all_combined_per_source"

# --- scripts/modeling/train/vlm_all_combined.py (added 2026-08-23, SCAFFOLDED — not yet trained) ---
#
# A third candidate architecture, kept as its own fully separate pipeline —
# same convention every other architecture in this project already follows
# (per-frame/windowed/image are deliberately never-shared, see e.g.
# src/image_dataset.py's module docstring). An earlier draft of this section
# tried a switchable "MODEL_APPROACH"/model_factory.py that picked between
# "vlm" and "conv" at runtime, since both read the exact same
# src/image_dataset.py ImageWindowedDataset input — reverted per explicit
# user request in favor of this project's normal one-architecture-per-file
# convention instead. scripts/modeling/train/image_windowed_all_combined.py
# (the "conv" approach — MobileNetV3+LSTM) is untouched by any of this.
#
# Deviation from the original task spec worth flagging explicitly: every
# model in this project (SpeakingDetectorRNN, WindowedSpeakingDetectorRNN,
# ImageWindowedSpeakingDetectorRNN) returns ONE logit per sample
# (BCEWithLogitsLoss, sigmoid at eval time) instead of 2-class softmax
# logits — VLMASDClassifier follows that same convention (forward() returns
# shape (B,), not (B, 2)) for drop-in compatibility with every existing
# training-loop/metrics helper (src/metrics.py's compute_metrics expects a
# sigmoid probability, masked_bce_loss, etc.) rather than introducing a
# second loss/metrics convention alongside the project's existing one.
#
# VERIFIED 2026-08-23 (web search — Qwen3.5-4B postdates this project's own
# knowledge, so this was checked live rather than assumed): Qwen/Qwen3.5-4B
# is a real released model (unified vision-language, hidden_size=2560,
# 32 layers) — but it is a chat/generation-oriented multimodal LM
# (transformers.AutoModelForMultimodalLM + AutoProcessor, real usage is
# building interleaved image+text chat turns and calling model.generate()),
# NOT a plain vision encoder with a clean pixel_values->pooler_output/
# last_hidden_state contract. An EARLIER version of this section/
# src/vlm_model.py assumed the latter (AutoModel + pixel_values +
# mean/attention/cls_token pooling + a separate linear head,
# VLM_HEAD_POOLING/VLM_HEAD_HIDDEN_DIM/VLM_CLASSIFICATION_HEAD_NUM_CLASSES
# below) — WRONG for this specific model, reverted per explicit user
# decision: "prompted classification" instead (see src/vlm_model.py's
# module docstring for the full design). VLMASDClassifier now builds one
# chat-style prompt per window ("does this person appear to be speaking?"),
# runs ONE forward pass (not generate(), to keep this differentiable for
# LoRA/full/partial-unfreeze training), and reads off the logit difference
# between a "Yes" and "No" answer token at the next-token position — see
# VLM_PROMPT_TEXT/VLM_YES_TOKEN_STRING/VLM_NO_TOKEN_STRING below.
# Deliberately OUTSIDE PROJECT_ROOT/data/, unlike this project's other
# caches (IMAGE_RESIZED_CACHE_DIR, WINDOWED_CACHE_DIR) — those live under
# data/ because the whole data/ folder is routinely moved out to
# /data/bola/{data,models} so the user can download a much-smaller project
# folder (see project memory), and a ~9GB model cache doesn't belong caught
# up in that shuffle. Root cause this exists at all: transformers/
# huggingface_hub default to ~/.cache/huggingface, which sits on THIS BOX'S
# root filesystem (98GB, found nearly full — 869MB free — 2026-08-23) rather
# than the /data mount (3.5TB, 1.9TB free) the rest of this project already
# uses; every VLM entry point must set HF_HOME to this before importing
# transformers/huggingface_hub (both read it once, at import time), or a
# real run will try to re-download the ~9GB model straight onto the nearly-
# full root disk.
VLM_HF_CACHE_DIR = "/data/bola/hf_cache"

# SWITCHED TO "Qwen/Qwen3-VL-4B-Instruct" then BACK to "Qwen/Qwen3.5-4B",
# both 2026-08-23 (both explicit user request):
#
# The Qwen3-VL detour: profiling proved Qwen3.5's per-layer forward memory
# (~1.2-2GB/layer, the actual OOM cause all day) was specific to its hybrid
# Gated-DeltaNet architecture. Qwen3-VL-4B-Instruct (same scale, 4.44B vs
# 4.56B params, standard Grouped Query Attention throughout, no
# hybrid/custom linear-attention layers) measured ~0.009GB/layer instead
# (~150-200x less) in isolated forward-only profiling. But once training
# conditions were measured correctly (LoRA + real backward + .train()
# actually called — an earlier bug meant NONE of that day's standalone
# tests had really exercised gradient checkpointing on either model).
# apples-to-apples per-window throughput came out essentially identical
# between the two models (0.68s/window Qwen3-VL vs 0.65s/window Qwen3.5) —
# the raw forward-only memory advantage didn't translate into a real
# training-time speed advantage once LoRA/backward/optimizer costs were
# included on both sides. 4-bit quantization was also tried on Qwen3-VL
# specifically because it's a standard (non-hybrid) architecture — real
# weight savings confirmed (8.88GB -> 4.74GB) but it still didn't fix
# memory at batch=1, for a different reason than Qwen3.5 (dequantize-
# on-the-fly overhead during forward, inherent to how PEFT does 4-bit
# QLoRA on ANY backend, not specific to bitsandbytes or this model).
#
# Reverted back to Qwen3.5-4B — explicit user judgment that it's simply
# the better model overall (a capability/quality call, not something this
# session's memory/speed profiling measures or overrides). The Qwen3-VL
# investigation is kept above as real, verified history in case it's
# useful context later — see src/vlm_model.py's module docstring for the
# fuller account of both models' real profiling.
VLM_PRETRAINED_NAME = "Qwen/Qwen3.5-4B"
VLM_NUM_FRAMES = 50  # matches WINDOW_SIZE (2s @ DATASET_FPS) exactly — see note above
VLM_FRAME_SAMPLING = "uniform"  # "uniform" (subsample to VLM_NUM_FRAMES if given more) | "all" (use every frame as-is)
VLM_FINETUNE_MODE = "lora"  # "lora" | "full" | "partial_unfreeze"

# 4-bit quantization (QLoRA) — explicit on/off toggle, 2026-08-23, explicit
# user request. Loads the ~4.56B frozen base weights in 4-bit
# (bitsandbytes) instead of bf16 — roughly a further ~4x reduction on top
# of bf16's own 2x-vs-fp32 saving for the base weights specifically
# (~9.1GB bf16 -> ~2.3GB 4-bit, before double-quant's further small
# reduction) — LoRA adapters (the actually-trainable params) stay in full
# precision on top, same as every other finetune_mode. Only compatible
# with VLM_FINETUNE_MODE="lora" — 4-bit weights aren't directly
# fine-tunable via full/partial-unfreeze backprop, only LoRA adapters on
# top of a frozen quantized base (the actual definition of "QLoRA") are
# supported; src/vlm_model.py raises a clear error if this is True under
# any other finetune_mode rather than silently doing something wrong.
# Requires a CUDA device — bitsandbytes 4-bit has no meaningful CPU path.
#
# DEFAULT SET False, 2026-08-23 (explicit user request), after real testing:
# quantization only compresses WEIGHT storage (~9.1GB bf16 -> 6.07GB 4-bit
# here, verified — 346 real bnb.nn.Linear4bit layers, correctly built), but
# the actual OOM at this model/sequence-length is dominated by per-layer
# ACTIVATION memory during the forward pass itself (~1.2-2GB/layer x 32
# layers, confirmed via hooked profiling to be IDENTICAL with gradient
# checkpointing on vs. off — so not a backward-retained-activations issue
# either). Quantization never targets that pool, so it didn't fix the OOM
# at VLM_BATCH_SIZE=1 despite working exactly as designed. Left here, fully
# wired and toggleable, in case it's useful later (e.g. if VLM_NUM_FRAMES
# or the sequence length this produces ever changes enough that weight
# memory becomes the binding constraint again).
VLM_USE_QUANTIZATION = False

# Only 4-bit is actually wired up in src/vlm_model.py right now — kept as
# its own named constant (rather than folding "4" into VLM_USE_QUANTIZATION's
# name) for documentation clarity and in case 8-bit support is ever added
# alongside it.
VLM_QUANTIZATION_BITS = 4

# The remaining four are BitsAndBytesConfig's real, standard QLoRA
# parameters — passed straight through in src/vlm_model.py's
# _build_quantization_config(), not reinvented:
VLM_QUANTIZATION_4BIT_QUANT_TYPE = "nf4"  # "nf4" | "fp4" — NormalFloat4 is the modern standard for LLM/QLoRA quantization, calibrated to how pretrained weights are actually distributed (roughly Gaussian) rather than a plain linear fp4 range
VLM_QUANTIZATION_4BIT_COMPUTE_DTYPE = "bfloat16"  # dtype used for the actual matmuls even though weights are STORED in 4-bit — matches this project's existing bf16 convention elsewhere in the VLM pipeline
VLM_QUANTIZATION_4BIT_USE_DOUBLE_QUANT = True  # nested quantization of the quantization constants themselves — standard QLoRA-paper default, a further small memory reduction (~0.4 bits/param) for negligible accuracy cost
VLM_QUANTIZATION_4BIT_QUANT_STORAGE = "uint8"  # the dtype the packed 4-bit weights are physically stored as — bitsandbytes' own default, essentially never needs changing from this

# Explicit on/off toggle (2026-08-23, explicit user request), separate from
# just implying "on" via config's presence — recomputes each of the 32
# decoder layers' activations during backward() instead of storing all of
# them simultaneously, trading ~20-30% more compute per step for
# significantly less activation memory. Applied via
# self.model.gradient_checkpointing_enable() in src/vlm_model.py — harmless
# to leave True during evaluation too (no backward() ever runs there, and
# HF's checkpoint implementation is a no-op under torch.no_grad()).
#
# DEFAULT FLIPPED True -> False -> True again, all 2026-08-23, same day:
# a real run with True (batch=4, OLD per-sample loop) confirmed
# checkpointing DID fix the original OOM, using only ~17GB/47GB GPU
# headroom — but throughput was ~15s/micro-batch (~21 DAYS/epoch), traced
# to the per-sample loop. forward() was rewritten the same day to do one
# real batched model() call per step (see src/vlm_model.py's "THROUGHPUT"
# section) — at that point this was flipped to False (explicit user
# request) to trade the spare memory margin back for speed, on the
# hypothesis that real batching wouldn't need it as much. CONFIRMED WRONG
# by an actual run, same day: real-batched forward at VLM_BATCH_SIZE=4
# with this False OOM'd again — this time inside one of the model's
# Gated-DeltaNet linear-attention layers (torch_chunk_gated_delta_rule),
# not the earlier LoRA MLP layer, but a real OOM either way. Batching
# clearly does NOT reduce peak activation memory the way it was hoped to
# — it processes all B windows' full activations simultaneously, same
# order of magnitude as the old loop kept alive at once, just computed
# together instead of sequentially. Flipped back to True. Real batching
# (kept) should still give most of its speed benefit even with
# checkpointing's recompute overhead layered back on top, since batching
# and checkpointing address different things (GPU parallelism vs. memory) —
# see src/vlm_model.py's docstring for the measured result once available.
#
# FLIPPED BACK TO False, 2026-08-23 (explicit user request, same day):
# decisive A/B profiling (real hooked memory trajectory, per decoder layer)
# showed checkpointing on vs. off produces an IDENTICAL per-layer memory
# growth curve (~1.2-2GB/layer x 32 layers) — ALL of that testing was on
# Qwen3.5-4B, since replaced (see VLM_PRETRAINED_NAME's comment).
#
# FLIPPED BACK TO True, 2026-08-23, same day, after switching to
# Qwen3-VL-4B-Instruct AND finding a real bug in the verification
# methodology itself: none of the day's standalone test scripts had ever
# called .train() before a forward pass (only scripts/modeling/train/
# vlm_all_combined.py's real run_epoch() does, via model.train(train_mode)
# — which was ALWAYS correct). Without an explicit .train() call, HF's
# gradient-checkpointing dispatch (gated on self.training internally)
# silently never activated in ANY test today, on EITHER model — explaining
# why checkpointing appeared to do nothing even for a standard, non-hybrid
# architecture. With .train() correctly called, checkpointing genuinely
# works for Qwen3-VL: real batch=8 step used only 11.32GB (free=15.39GB
# after), vs. batch=1 alone needing 31.66GB without .train() ever having
# been called. This means the EARLIER "checkpointing does nothing" finding
# for Qwen3.5 is also unproven, not necessarily true — it was never
# re-tested with .train() before the model switch, so treat that earlier
# conclusion as historical/uncertain rather than settled.
#
# FLIPPED BACK TO False again, 2026-08-23, same day, after a real
# WALL-CLOCK comparison at the effective-batch-32 target (explicit user
# request): batch=8+checkpointing+accumulation=4 measured 8.08s/micro-batch
# (4 steps x 8.08s = 32.3s per effective batch of 32) vs.
# batch=1+no-checkpointing+accumulation=32 measured 0.68s/micro-batch (32
# steps x 0.68s = 21.8s per effective batch) — checkpointing's real
# recompute overhead at batch=8 outweighs the benefit of fewer/bigger
# steps, so batch=1+no-checkpointing is actually FASTER here despite
# needing 8x more steps. Also resolves the earlier "is Qwen3-VL slower
# than Qwen3.5?" question: apples-to-apples (both batch=1, no
# checkpointing), the two models measured 0.68s/window vs 0.65s/window —
# essentially identical, confirming the earlier Qwen3.5 throughput numbers
# were never really paying checkpointing's cost either.
#
# NOTE: VLM_PRETRAINED_NAME reverted back to Qwen3.5-4B after all of the
# above (explicit user request, model-quality judgment) — this value
# (False) carries over cleanly: it's also exactly what Qwen3.5's own
# earlier, independent real verification used (batch=1, no checkpointing,
# 38.84GB peak, real forward+backward+optimizer.step() succeeded).
VLM_USE_GRADIENT_CHECKPOINTING = False

# Explicit on/off toggle (2026-08-23, explicit user request) for
# scripts/modeling/train/vlm_all_combined.py's gradient-accumulation loop —
# when False, behaves exactly like VLM_GRADIENT_ACCUMULATION_STEPS=1 (step
# the optimizer every micro-batch) regardless of that constant's value,
# without needing to actually change/lose the configured step count. See
# VLM_GRADIENT_ACCUMULATION_STEPS below for what accumulation itself does.
VLM_USE_GRADIENT_ACCUMULATION = True

# The chat-turn text asked alongside the window's VLM_NUM_FRAMES images —
# deliberately phrased for a short, near-single-token answer (see
# src/vlm_model.py's _answer_token_ids()). Untested against the real model's
# actual chat template/tokenizer — the exact phrasing may need adjusting
# once that's verified (e.g. if "Yes"/"No" don't tokenize as expected single
# tokens, or the model responds with different wording by default).
VLM_PROMPT_TEXT = (
    "Look at this sequence of video frames showing one person's face. "
    "Is this person speaking in these frames? Answer with only Yes or No."
)
VLM_YES_TOKEN_STRING = "Yes"
VLM_NO_TOKEN_STRING = "No"

# Deliberately NOT IMAGE_BATCH_SIZE (64) — a 4B-parameter backbone run
# VLM_NUM_FRAMES=50 times per window makes a large windows/batch count an
# easy way to OOM even on this box's GPUs.
#
# HISTORY (condensed — the full blow-by-blow lived here when
# VLM_PRETRAINED_NAME was Qwen3.5-4B): that model's hybrid Gated-DeltaNet
# layers forced this all the way down to VLM_BATCH_SIZE=1 even after
# installing kernels/flash-linear-attention (see requirements.txt) — see
# git history / earlier session notes for the full sequence (4->3->2->1,
# quantization not helping, gradient checkpointing appearing not to help).
#
# CURRENT VALUE, 2026-08-23, after switching to Qwen3-VL-4B-Instruct (see
# VLM_PRETRAINED_NAME's comment) AND fixing a real bug in how this was
# being tested (see VLM_USE_GRADIENT_CHECKPOINTING's comment — standalone
# verification scripts never called .train(), so checkpointing silently
# never activated in ANY test that day). With .train() correctly called
# and checkpointing genuinely active, a real batch-size sweep gave: batch=1
# 3.13GB, batch=2 2.55GB, batch=4 5.62GB, batch=8 11.32GB, batch=16 OOM'd —
# batch=8 was chosen from that sweep first.
#
# LOWERED BACK TO 1, same day, after a real WALL-CLOCK comparison at the
# effective-batch-32 target (explicit user request): batch=8+checkpointing
# measured SLOWER overall (32.3s per effective batch of 32) than
# Raised 1->2, 2026-08-23, after device_map="auto" (2-GPU sharding, see
# VLM_PRETRAINED_NAME's loading in src/vlm_model.py) made more memory
# available — empirically swept 1/2/3/4 with a real forward+backward+
# optimizer.step() (logits_to_keep=1 fix already applied): batch=1 leaves a
# healthy 22.2GB free margin on the tighter of the two devices; batch=2
# works but only leaves ~1.77GB free on device 1 (thin — each window is
# ~3,338 tokens after image tokenization, and self-attention memory scales
# with batch_size * seq_len^2, so this is a real, expected jump, not a
# bug); batch=3 and batch=4 both OOM outright. Lowered back 2->1,
# 2026-08-26, explicit user request — reverts to the safe margin given
# batch=2's thin ~1.77GB headroom on a shared box where an unrelated
# always-on vllm serve process (reserves ~50% of each GPU) was directly
# observed spiking further during earlier testing.
#
# RAISED 1->4, 2026-08-27: both H200s verified completely empty (0 MiB of
# 143,771 MiB each) — the always-on `vllm serve` container that motivated
# the 2->1 revert above is stopped, returning ~59GB per GPU, and the
# attendance pipeline's ~23GB on GPU 0 is gone too. Budget, extrapolated
# from the two REAL measurements above (batch=1 -> 22.2GB free on the
# tighter device, batch=2 -> 1.77GB free, so ~20.4GB marginal cost per
# extra window): with vLLM's ~59GB returned, batch=1 now has ~81GB free,
# putting batch=4 at ~20GB margin — the same safety margin batch=1 had
# under the old shared-box conditions. batch=5 extrapolates to ~0GB, so 4
# is the ceiling that still keeps real headroom.
#
# NOT YET EMPIRICALLY VERIFIED — this is a 2-point linear extrapolation,
# not a measured sweep. The marginal term is expected to be linear
# (self-attention memory ~ batch_size * seq_len^2 at fixed seq_len), but
# re-run the real forward+backward+optimizer.step() sweep on the training
# box before trusting it, exactly as the 1/2/3/4 sweep above did. If it
# OOMs, step down to 3 rather than straight back to 1.
#
# NOTE on the "batch=8 measured SLOWER" result above: that run had
# gradient checkpointing ON. Checkpointing recomputes activations during
# backward and is itself typically 30-40% slower, so that measurement
# confounds batch size with checkpointing and is NOT evidence against
# raising the batch here — VLM_USE_GRADIENT_CHECKPOINTING is False.
VLM_BATCH_SIZE = 4

# Effective batch size = VLM_BATCH_SIZE * VLM_GRADIENT_ACCUMULATION_STEPS
# (1 * 32 = 32, the effective batch this project has targeted throughout
# the VLM work) — scripts/modeling/train/vlm_all_combined.py's run_epoch()
# accumulates gradients (loss scaled by 1/this, backward() every
# micro-batch, optimizer.step()+zero_grad() only every this-many
# micro-batches or at the epoch's last batch) instead of stepping every
# micro-batch. Restored 16->32, 2026-08-26, alongside VLM_BATCH_SIZE's
# 2->1 revert, to keep the effective batch at 32 (1 * 32 = 32).
# Lowered 32->8, 2026-08-27, alongside VLM_BATCH_SIZE's 1->4, to HOLD the
# effective batch at 32 (4 * 8 = 32). Leaving this at 32 would have made
# the effective batch 128, silently changing the optimization rather than
# just speeding it up — VLM_LORA_LR was chosen at effective-batch-32, so
# that would be a different experiment, not the same run made faster.
VLM_GRADIENT_ACCUMULATION_STEPS = 8

# Own epoch cap — separate from the shared NUM_EPOCHS (200, used by every
# other training script: conv/landmark/image) so capping the VLM run
# doesn't silently also cap those other pipelines. Set to 1 (2026-08-23,
# explicit user request) after deciding a single epoch over more unique
# per-source samples beats repeating a smaller sample across multiple
# epochs (see VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE's own reasoning) —
# VLM_EARLY_STOPPING_PATIENCE=2 simply never gets the chance to fire at
# this value, which is fine, not a conflict.
VLM_NUM_EPOCHS = 1

# Separate from IMAGE_EARLY_STOPPING_PATIENCE (3) — a VLM epoch is expected
# to cost even more than the image/conv pipeline's already-slow ~20min/epoch
# (same "checkpoint selection already protects against post-best overfitting,
# so a high patience only wastes compute" reasoning as IMAGE_EARLY_STOPPING_PATIENCE's
# own comment, just more extreme here — at ~3.7 days/epoch, even patience=3
# risks ~11 wasted days past the real best epoch).
# Lowered 3->2, 2026-08-23, explicit user request.
VLM_EARLY_STOPPING_PATIENCE = 2

VLM_LORA_R = 16
VLM_LORA_ALPHA = 32
VLM_LORA_DROPOUT = 0.05
# Verified 2026-08-23 against the real loaded model (Qwen/Qwen3.5-4B is a
# HYBRID architecture — see project memory): only 8 of the model's 32
# language-model layers are standard attention with q_proj/v_proj/o_proj;
# the other 24 are a Gated-DeltaNet-style linear-attention variant with
# entirely different projection names (in_proj_qkv/in_proj_z/in_proj_b/
# in_proj_a/out_proj, no overlap at all with q_proj/v_proj). The original
# ["q_proj", "v_proj"] therefore adapted ZERO parameters in 75% of the
# model's depth, not just "fewer" — a much bigger gap than the layer count
# alone suggests. Both layer types DO share identically-named
# mlp.gate_proj/up_proj/down_proj, so including those gives LoRA real
# coverage across all 32 layers regardless of attention type — added here
# along with o_proj (attention-only LoRA restricted to q/v is usually
# considered incomplete; o_proj is cheap to include on the 8 layers that
# have it). Deliberately NOT reaching into the Gated-DeltaNet layers'
# attention-specific projections (in_proj_*/out_proj) — different shapes/
# semantics from standard attention, would need separate handling, and the
# MLP-based whole-depth coverage below is a simpler, well-precedented win
# to start with.
VLM_LORA_TARGET_MODULES = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

VLM_PARTIAL_UNFREEZE_NUM_BLOCKS = 4  # only read when VLM_FINETUNE_MODE == "partial_unfreeze"

# Optimizer learning rates for scripts/modeling/train/vlm_all_combined.py's
# per-finetune_mode param group (see that script for how these are applied) —
# LoRA/head-only params train fastest since there are few of them and they
# start from random init; full fine-tuning of a pretrained VLM needs a much
# lower rate to avoid destroying pretrained weights, plus warmup.
VLM_LORA_LR = 1e-4
VLM_FULL_FINETUNE_LR = 2e-5
VLM_FULL_FINETUNE_WARMUP_STEPS = 500
VLM_PARTIAL_UNFREEZE_LR = 5e-5

# Explicit on/off toggle, 2026-08-23, explicit user request (prompted by
# seeing bitsandbytes' paged optimizers used in an unrelated SFTTrainer
# example) — bitsandbytes.optim.PagedAdamW32bit instead of plain
# torch.optim.Adam in scripts/modeling/train/vlm_all_combined.py. "Paged"
# means the optimizer state (momentum/variance, 2x the ~20.8M trainable
# param count here — small in absolute terms) uses CUDA unified memory and
# can be transparently paged out to CPU RAM under GPU memory pressure
# instead of hitting an OOM outright — real, relevant to this project given
# how close to the memory ceiling VLM_BATCH_SIZE=1 already sits. Not yet
# benchmarked against plain Adam here (paging adds host<->device transfer
# overhead if it actually triggers, so it could be neutral-to-cheap if
# there's enough headroom for it to never actually page, or a real slowdown
# if it does — untested either way).
VLM_USE_PAGED_OPTIMIZER = True

# Per-source subsample size for quick train/test smoke runs, wired through
# to src/image_dataset.py's ImageWindowedDataset(sample_size_per_source=...)
# (shared with the conv pipeline, opt-in via None so conv's own full-dataset
# runs are unaffected). None = use every window (real training default).
# A number N = randomly keep N windows from EACH of the 6 combined sources
# (N*6 total for train). Train/val/test are already disjoint by construction
# (separate CSV files/person_id tracks per split) so independently
# subsampling each split's own window pool cannot leak samples across
# splits. Explicit user request 2026-08-23: val and test each use N/5 per
# source (e.g. N=500 -> train 500*6=3000, val 100*6=600, test 100*6=600).
VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE = 10000
VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE = (
    VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE // 5 if VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE is not None else None
)

# Own checkpoint/log/eval filenames — separate from every other architecture
# in this project, same "never overwrite a different architecture's result"
# convention as IMAGE_ALL_WINDOWED_* above (though see
# [[missing-checkpoint-incident]]: CHECKPOINTS_DIR itself is still shared and
# reset_dir()-wiped at the start of every training script regardless of
# filename — distinct filenames don't protect a prior run's checkpoint from
# being deleted by a new one).
VLM_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME = "vlm_all_windowed_best_model.pt"  # gitignored (checkpoints/*.pt)
VLM_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME = "vlm_all_windowed_last_model.pt"  # gitignored (checkpoints/*.pt)
VLM_ALL_WINDOWED_TRAIN_METRICS_FILENAME = "vlm_all_windowed_train_metrics.csv"    # gitignored (logs/*.csv)
VLM_ALL_WINDOWED_TRAIN_LOG_FILENAME = "vlm_all_windowed_train.log"                # gitignored (logs/*.log)
VLM_ALL_WINDOWED_LOSS_CURVE_FILENAME = "vlm_all_windowed_loss_curve.png"          # gitignored (logs/*.png)
VLM_ALL_WINDOWED_ACCURACY_CURVE_FILENAME = "vlm_all_windowed_accuracy_curve.png"  # gitignored (logs/*.png)
VLM_ALL_WINDOWED_TRAINING_REPORT_FILENAME = "vlm_all_windowed_README.md"          # gitignored (logs/*.md)
VLM_ALL_WINDOWED_EVALUATION_DIR = PROJECT_ROOT / "evaluation_vlm_windowed_all_combined"
VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR = PROJECT_ROOT / "evaluation_vlm_windowed_all_combined_per_source"
