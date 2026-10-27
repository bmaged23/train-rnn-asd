# Prepared-window inference

This interface accepts **already preprocessed landmark feature windows**. It does
not extract faces from raw video. No trained checkpoint is distributed. Missing
weights cause an error; random initialization is never used as a fallback.
The bidirectional model needs a completed window, so this is not causal streaming.

## Install and run

From the repository root, use Python 3.13.16 on Linux x86-64:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt
python -m pip check
python -m scripts.inference.predict_windows --help
```

Obtain your saved windowed BiLSTM state dictionary and its exact training
configuration. Write a manifest with these fields (values below illustrate the
schema, **not a compatible configuration for an existing checkpoint**):

```json
{
  "schema_version": 1,
  "checkpoint_sha256": "SHA256_OF_YOUR_CHECKPOINT",
  "feature_names": ["feature_0", "feature_1"],
  "window_size": 4,
  "hidden_size": 3,
  "num_layers": 1,
  "dropout": 0,
  "threshold": 0.5,
  "preprocessing": "Document normalization, landmark order, geometry, MAR and padding from training"
}
```

Use `sha256sum model.pt` to obtain the hash. Feature names must describe the exact
ordered model input, including derived geometry features. Hash validation binds
the metadata to a weight file; it does not prove that metadata correctly describes
the original training preprocessing. Use only trusted checkpoint sources.

The input JSON contains `feature_names` matching the manifest, `features` shaped
`[batch, window_size, feature_count]`, and integer `lengths` (one per window).
The batch limit is 64. All values must be finite; lengths exclude right padding.
No normalization or landmark extraction happens inside this interface.

```bash
python -m scripts.inference.predict_windows \
  --checkpoint models/model.pt --manifest models/manifest.json \
  --input input/windows.json --output output/predictions.json
```

Create the output directory first. Results contain probabilities, thresholded
speaking labels and the checkpoint hash. Strict state-dictionary loading rejects
incompatible architectures. CUDA can be explicitly requested with `--device cuda`
in a CUDA-enabled PyTorch installation; this route has not been GPU-qualified.

## Container toolkit

The Docker recipe is a **CPU inference toolkit requiring your own weights and
prepared inputs**, not a ready-to-use trained detector or a published Docker Hub
release. It excludes datasets, models, caches, notebooks and credentials.

```bash
docker build -f deployment/docker/Dockerfile -t asd-window-toolkit:local .
docker run --rm asd-window-toolkit:local --help
mkdir -p output
chmod a+w output
docker run --rm \
  -v "$PWD/models:/models:ro" -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" asd-window-toolkit:local \
  --checkpoint /models/model.pt --manifest /models/manifest.json \
  --input /input/windows.json --output /output/predictions.json
```

The process runs as UID 10001. Ensure the output mount is writable; apply a
suitable ownership/ACL instead of broad write permission on shared machines.
No training or dataset download is performed. Actual quality/latency/VRAM results
remain unavailable without original trained weights and labeled inputs.

## Verification scope

`tests/test_inference.py` uses temporary random-weight checkpoints to verify
loading, hash checks, feature ordering, padding lengths, numerical output and
failure behavior. These are software tests and provide no accuracy evidence.

CI additionally builds the toolkit and invokes its actual packaged CLI in a
read-only, network-disabled container with temporary synthetic weights. It checks
that probabilities match native PyTorch and missing weights fail explicitly:

```bash
python -m scripts.validation.validate_inference_container --image asd-window-toolkit:local
```

This pipeline performs software CI and container verification. It does not train,
evaluate accuracy, publish an image or deploy a running service.
