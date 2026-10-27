# Repository map and maintenance

| Location | Responsibility |
|---|---|
| `config.py` | Research paths, feature settings and bounded worker pools |
| `src/serving/` | API application, schemas and serving settings |
| `src/inference.py` | Strict checkpoint/manifest loading and window prediction |
| `src/*model.py` | Existing landmark, image and VLM architectures |
| `src/*dataset.py` | Existing track/window/image datasets and provenance |
| `src/metrics.py` | Validated binary metrics and visualizations |
| `scripts/application/` | Start the API |
| `scripts/inference/` | Predict prepared windows from the terminal |
| `scripts/release/` | Publish evaluated toolkit and verify public pulls |
| `scripts/validation/` | Split audits and packaged CLI/HTTP checks |
| `scripts/dataset/` | Research source preparation/download commands |
| `scripts/features/` | Research feature extraction and cache generation |
| `scripts/splits/` | Research split preparation |
| `scripts/modeling/train/` | Existing training experiments; not run in this work |
| `scripts/modeling/evaluate/` | Existing trained-model evaluation entry points |
| `tests/` | Offline contracts and API regression tests |
| `requirements/` | Explicit dependency profiles |
| `deployment/docker/` | Pinned-base, non-root CPU serving image |
| `docs/api/`, `docs/inference/` | API protocol and checkpoint/input contract |
| `docs/deployment/`, `docs/quality/` | Delivery process and validation scope |
| `docs/assets/`, `notebooks/` | Preserved historical figures and research evidence |

Root `requirements.txt` selects the pinned API environment. `requirements/ci.txt`
adds test/lint dependencies; `cpu.txt` and `evaluation.txt` are shared pinned
subprofiles. `research-unqualified.txt` preserves the old combined research stack
for provenance, with no claim of a newly tested full training environment. Do not
install it just to use the API; it includes heavy GPU/VLM dependencies.

Keep checkpoints, raw datasets, prepared private inputs, generated predictions,
logs, caches, environments and local implementation notes ignored. The Docker
build uses an allowlist, so research notebooks/data/tests/private `.env` files
never enter the image. Preserve tracked historical figures and notebooks: they
explain past experiments and are not disposable temporary files.

Architectures and research entry points retain their established names/paths to
avoid breaking notebook imports. New serving, inference and delivery code has
dedicated descriptive folders. No original author attribution is removed.
