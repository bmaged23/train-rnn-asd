# Prepared-window HTTP API

The service loads one supplied windowed BiLSTM checkpoint at startup. It accepts
completed, already preprocessed landmark windows. There is no raw-video upload,
face extraction or streaming endpoint. This matches the model's bidirectional
architecture. No training or automatic model/data download occurs.

## Start from source

```bash
git clone https://github.com/bmaged23/train-rnn-asd.git
cd train-rnn-asd
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
cp .env.example .env
```

Use Python 3.13.16 on Linux x86-64. Put your trusted checkpoint and matching
[manifest](../inference/README.md) under ignored `models/`. In `.env`, set:

```dotenv
ASD_CHECKPOINT=models/model.pt
ASD_MANIFEST=models/manifest.json
ASD_DEVICE=cpu
ASD_HOST=127.0.0.1
ASD_PORT=8000
ASD_MAX_BODY_BYTES=4194304
ASD_API_TOKEN=your_private_token
```

Start the service:

```bash
python -m scripts.application.start_api
```

Interactive documentation: http://127.0.0.1:8000/docs. The JSON specification is
http://127.0.0.1:8000/openapi.json. Swagger's page uses external UI assets; the
JSON specification and prediction service work without those assets.

## Endpoints

| Method / path | Purpose | Model required | Token when configured |
|---|---|---|---|
| `GET /health/live` | Web process is responding | No | No |
| `GET /health/ready` | Valid checkpoint is loaded; includes its hash/device | Yes | No |
| `GET /v1/model` | Exact manifest, architecture and ordered input features | Yes | Yes |
| `POST /v1/predict` | Probabilities and thresholded speaking labels | Yes | Yes |
| `GET /metrics` | Process-local counters and inference duration | No | Yes |
| `GET /docs`, `GET /openapi.json` | Interactive documentation / schema | No | No |

The HTTP process may be live while readiness and prediction return **503**. This
happens when checkpoint/manifest paths are absent, files are missing, hashes differ
or state dictionaries are incompatible. Fix the files/configuration and restart.
Only a valid supplied checkpoint makes the service ready; no fallback weights.

## Prediction request

Save your input as `input/windows.json`:

```json
{
  "feature_names": ["feature_0", "feature_1"],
  "features": [[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.0, 0.0]]],
  "lengths": [3]
}
```

This is a schema example with two features and four timesteps, not the real
training feature configuration. Names/order and dimensions must match your
manifest exactly. `lengths` excludes right padding. Values must be finite numbers;
one request supports 1–64 windows and at most the configured body byte limit.

```bash
# Set ASD_API_TOKEN in your shell to the value configured for the server.
curl --fail-with-body http://127.0.0.1:8000/v1/predict \
  -H "Authorization: Bearer $ASD_API_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @input/windows.json
```

The response contains `probabilities`, `speaking`, `threshold` and
`checkpoint_sha256`. Probabilities are model scores, not evidence of calibration.
There is no accuracy estimate in an individual prediction response.

| Status | Meaning |
|---|---|
| 200 | Request succeeded |
| 401 | Bearer token missing/incorrect, when enabled |
| 413 | Actual received body exceeds byte limit, including chunked input |
| 422 | JSON/schema, feature order, shape or lengths are invalid |
| 503 | Model unavailable, or inference busy (`Retry-After: 1` for busy) |
| 500 | Model execution failed; internal details are withheld |

## Runtime limits and monitoring

The launcher uses one Uvicorn worker, a server concurrency limit of 16 and a
five-second keepalive timeout. Inference is serialized with a nonblocking lock:
a second inference request receives retryable 503 rather than an unbounded queue.
Health endpoints remain independent of the inference worker. One worker means
one model copy; increasing web workers would duplicate it and its memory.

`/metrics` returns `completed`, `rejected_input`, `busy`, `failed` and accumulated
`inference_seconds`. These cover prediction-handler execution, not request parsing,
authentication failures or the whole device. Counters reset on process restart.
This is a JSON diagnostic endpoint, not a Prometheus exporter or resource sampler.

Bind to loopback for local use. The optional token protects predictions, metadata
and metrics; no token means these endpoints are unauthenticated. The service has
no TLS, account management or internet-facing rate limiter. A public deployment
requires an appropriate proxy/access policy. Native CUDA can be explicitly
requested with a compatible CUDA PyTorch installation; the published image targets
CPU and CUDA serving has not been qualified in this work.

## Verification

Eight HTTP regression tests use temporary synthetic checkpoints. Container checks
also make real HTTP requests to a read-only, non-root container: native prediction
parity, readiness, missing-model rejection, authentication, input rejection,
counters and OpenAPI coverage. These establish service behavior, not ASD accuracy.

See the [Docker and delivery guide](../deployment/README.md).
