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

# curl/wget intentionally not installed: healthcheck uses Python.

WORKDIR /app

# Install CPU-only torch first (smaller image, no NVIDIA wheels).
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

COPY pyproject.toml ./
RUN pip install \
        "fastapi==0.115.6" \
        "uvicorn[standard]==0.32.1" \
        "sentence-transformers==3.3.1" \
        "pydantic==2.10.3"

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
