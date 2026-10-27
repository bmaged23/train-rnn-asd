# Checkpoint-required CPU inference toolkit, not a pretrained detector.
FROM python:3.13.16-slim@sha256:cf50a4f6ac4f026458b08d5222023d700ac806410dff5e9ca8728fa44f2b850f
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib ASD_WORKER_LIMIT=2 ASD_HOST=0.0.0.0 ASD_PORT=8000
WORKDIR /app
LABEL org.opencontainers.image.source="https://github.com/bmaged23/train-rnn-asd" org.opencontainers.image.licenses="MIT"
COPY requirements/evaluation.txt requirements/cpu.txt requirements/api.txt /app/requirements/
RUN python -m pip install --no-cache-dir -r requirements/api.txt && python -m pip check
COPY config.py /app/config.py
COPY LICENSE /app/LICENSE
COPY src /app/src
COPY scripts/inference /app/scripts/inference
COPY scripts/application /app/scripts/application
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]
ENTRYPOINT ["python"]
CMD ["-m", "scripts.application.start_api"]
