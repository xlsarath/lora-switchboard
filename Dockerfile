# syntax=docker/dockerfile:1

# ---- build: produce a wheel -------------------------------------------------
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.2.2
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="lora-switchboard" \
      org.opencontainers.image.description="Safe LoRA adapter rollouts for vLLM" \
      org.opencontainers.image.source="https://github.com/xlsarath/lora-switchboard" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LSB_DB_PATH=/data/switchboard.db

COPY --from=build /dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl \
 && groupadd --system --gid 10001 switchboard \
 && useradd --system --uid 10001 --gid switchboard --home-dir /app --shell /usr/sbin/nologin switchboard \
 && mkdir -p /data /app \
 && chown switchboard:switchboard /data

WORKDIR /app
COPY --chown=switchboard:switchboard examples ./examples

USER 10001:10001
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]

ENTRYPOINT ["lora-switchboard"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
