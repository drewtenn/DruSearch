FROM python:3.11-slim

RUN --mount=type=bind,from=host_ca,source=.,target=/host_ca,ro \
    if [ -d /host_ca ] && ls /host_ca/*.crt >/dev/null 2>&1; then \
      cp /host_ca/*.crt /usr/local/share/ca-certificates/ && update-ca-certificates; \
    fi

ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Healthcheck uses Python urllib; no extra apt packages needed.

RUN pip install --no-cache-dir \
        mlflow==2.18.0 \
        boto3==1.35.79 \
        psycopg2-binary==2.9.10

WORKDIR /mlflow
EXPOSE 5000
