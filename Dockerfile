FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SCOUTRAG_ANSWER_MODE=template \
    SCOUTRAG_ENABLE_DENSE_RETRIEVAL=false \
    SCOUTRAG_ENVIRONMENT=production \
    SCOUTRAG_LOCAL_FILES_ONLY=true

WORKDIR /app

RUN addgroup --system scoutrag \
    && adduser --system --ingroup scoutrag --home /app scoutrag

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data/processed/bundesliga-2023-2024/player_season_profiles.parquet \
    data/processed/bundesliga-2023-2024/player_season_profiles.parquet
COPY data/processed/bundesliga-2023-2024/player_metric_evidence.parquet \
    data/processed/bundesliga-2023-2024/player_metric_evidence.parquet

RUN python -m pip install --no-cache-dir .

USER scoutrag
EXPOSE 8000

CMD ["sh", "-c", "uvicorn scoutrag.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
