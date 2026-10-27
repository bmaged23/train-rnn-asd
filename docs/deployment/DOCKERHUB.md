# Active Speaker Detection — checkpoint-required inference API

[GitHub source, historical model results and documentation](https://github.com/bmaged23/train-rnn-asd)
· [API guide](https://github.com/bmaged23/train-rnn-asd/blob/master/docs/api/README.md)
· [Docker and CI/CD](https://github.com/bmaged23/train-rnn-asd/blob/master/docs/deployment/README.md)

CPU service for a supplied windowed BiLSTM checkpoint. Predicts visual speaking
status from completed, preprocessed landmark feature windows. No raw-video upload,
training, pretrained checkpoint or automatic model/dataset download is included.
Linux x86-64 Docker is the tested target. No GPU is required by this image.

## Pull and run

Supply `models/model.pt` and a matching `models/manifest.json` using the
[checkpoint/input schema](https://github.com/bmaged23/train-rnn-asd/blob/master/docs/inference/README.md).
Create a private `.env` with `ASD_API_TOKEN=your_private_token` if desired.

```bash
docker pull bmaged23/train-rnn-asd:cpu
docker run --rm --read-only --tmpfs /tmp --env-file .env \
  -p 127.0.0.1:8000:8000 -v "$PWD/models:/models:ro" \
  -e ASD_CHECKPOINT=/models/model.pt -e ASD_MANIFEST=/models/manifest.json \
  bmaged23/train-rnn-asd:cpu
```

Open http://127.0.0.1:8000/docs. Use `POST /v1/predict` for JSON feature windows;
`GET /v1/model` describes the exact feature order. `GET /metrics` exposes
process-local prediction counters/timing. No Git clone or extra dependency install
is needed. The image runs as UID 10001; model files must be readable.

Readiness: http://127.0.0.1:8000/health/ready. Without a valid checkpoint and
manifest, the process reports 503 and the container is unhealthy. It never falls
back to random weights. Stop with Ctrl+C.

## Verification and delivery

34 offline tests cover software/model contracts and API behavior. The candidate
image runs actual CLI and HTTP inference with tiny temporary synthetic checkpoints,
checks parity against native PyTorch and validates readiness, errors and auth.
The published immutable image is pulled anonymously and tested again before the
`cpu` tag is promoted. Runtime pushes automate these gates and publication through
GitHub Actions. Original trained-model accuracy is not re-evaluated by these tests.

The image excludes weights, research datasets, personal `.env` files and caches.
This is a toolkit for supplied checkpoints, not a ready-trained ASD detector.
