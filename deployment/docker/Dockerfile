# Checkpoint-required CPU inference toolkit, not a pretrained detector.
FROM python:3.13.16-slim@sha256:cf50a4f6ac4f026458b08d5222023d700ac806410dff5e9ca8728fa44f2b850f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib ASD_WORKER_LIMIT=2
WORKDIR /app
COPY requirements /app/requirements
RUN python -m pip install --no-cache-dir -r requirements/cpu.txt && python -m pip check
COPY config.py /app/config.py
COPY src /app/src
COPY scripts/inference /app/scripts/inference
USER 10001:10001
ENTRYPOINT ["python", "-m", "scripts.inference.predict_windows"]
