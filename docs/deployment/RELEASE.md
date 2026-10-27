# Verified checkpoint-required CPU toolkit release

Verified on 27 October 2026.

| Evidence | Result |
|---|---|
| Runtime source | `b24353dffe17bc95f239a8d521b09539565a4c6b` |
| GitHub CI/CD | [40434951510](https://github.com/bmaged23/train-rnn-asd/actions/runs/40434951510), all stages passed in 2m54s |
| Public image | `bmaged23/train-rnn-asd:cpu` |
| Immutable image | `bmaged23/train-rnn-asd@sha256:ec2d7420bae85afba4dc7a5ade80842f2abf6955fa11529ac1f8b1956938c753` |
| Platform | Linux amd64, CPU PyTorch 2.10.0, Python 3.13.16 |
| Registry compressed size | 472,417,137 bytes (about 450.5 MiB); local unpacked storage differs |
| Offline suite | 34/34 passed locally and in GitHub Actions |
| Candidate CLI | Synthetic-weight predictions match native PyTorch; missing weights rejected |
| Candidate HTTP | Actual prediction parity, readiness, auth, invalid inputs, metrics and OpenAPI passed |
| Public distribution | Anonymous immutable pull followed by repeated CLI/HTTP container checks passed before `cpu` promotion |
| Public overview | Anonymous verification matched the checked-in Hub README, including GitHub/API/deployment links |

Local checks used tiny temporary synthetic checkpoints, which were not included
in the image. The HTTP service ran as a non-root process in a read-only container;
the CLI check additionally disabled networking. Missing checkpoints caused explicit
CLI failure and HTTP 503 readiness. No random-weight fallback exists in serving.

No original trained model was available. These are software/container results,
not ASD accuracy, calibrated confidence, model latency or resource benchmarks.
Historical notebook metrics remain historical. The owner excluded new training
and large dataset downloads; GPU serving and raw-video inference are unqualified.

Reproduce the released toolkit exactly:

```bash
docker pull bmaged23/train-rnn-asd@sha256:ec2d7420bae85afba4dc7a5ade80842f2abf6955fa11529ac1f8b1956938c753
```

Supply your checkpoint/manifest using the [run instructions](README.md).
Documentation changes after the runtime source above do not rebuild this image.
