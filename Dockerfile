FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NBA_MODEL_PATH=/app/artifacts/home_win_lean_logistic.joblib \
    NBA_MODEL_METADATA_PATH=/app/artifacts/metrics_lean_logistic.json \
    NBA_FINAL_EVALUATION_PATH=/app/artifacts/final_holdout_corrected/holdout_metrics.json

WORKDIR /app

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install .

COPY --chown=10001:10001 artifacts/home_win_lean_logistic.joblib ./artifacts/home_win_lean_logistic.joblib
COPY --chown=10001:10001 artifacts/metrics_lean_logistic.json ./artifacts/metrics_lean_logistic.json
COPY --chown=10001:10001 artifacts/final_holdout_corrected/holdout_metrics.json ./artifacts/final_holdout_corrected/holdout_metrics.json

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "nba_predictor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
