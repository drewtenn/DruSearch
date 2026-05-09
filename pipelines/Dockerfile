# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=bind,from=host_ca,source=.,target=/host_ca,ro \
    if [ -d /host_ca ] && ls /host_ca/*.crt >/dev/null 2>&1; then \
      cp /host_ca/*.crt /usr/local/share/ca-certificates/ && update-ca-certificates; \
    fi

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN for i in 1 2 3; do \
      apt-get update && \
      apt-get install -y --no-install-recommends build-essential git libgomp1 && \
      rm -rf /var/lib/apt/lists/* && break || sleep 5; \
    done

WORKDIR /app

# CPU-only torch first to keep image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu

COPY pyproject.toml ./
RUN pip install \
        "psycopg[binary,pool]==3.2.3" \
        "opensearch-py==2.8.0" \
        "redis==5.2.1" \
        "boto3==1.35.79" \
        "mlflow==2.18.0" \
        "pandas==2.2.3" \
        "pyarrow==18.1.0" \
        "numpy==2.1.3" \
        "scikit-learn==1.6.0" \
        "lightgbm==4.5.0" \
        "xgboost==2.1.3" \
        "sentence-transformers==3.3.1" \
        "tqdm==4.67.1" \
        "tenacity==9.0.0" \
        "python-dotenv==1.0.1" \
        "httpx==0.28.1"

COPY pipelines ./pipelines

CMD ["python", "-m", "pipelines.check_health"]
