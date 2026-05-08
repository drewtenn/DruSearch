# syntax=docker/dockerfile:1.7
FROM golang:1.25-bookworm AS build

# Trust any extra CA certificates from the host (corp proxies, etc.).
RUN --mount=type=bind,from=host_ca,source=.,target=/host_ca,ro \
    if [ -d /host_ca ] && ls /host_ca/*.crt >/dev/null 2>&1; then \
      cp /host_ca/*.crt /usr/local/share/ca-certificates/ && update-ca-certificates; \
    fi

WORKDIR /src

# Cache module downloads.
COPY go.mod go.sum* ./
RUN go mod download || true

COPY . .
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

FROM gcr.io/distroless/base-debian12:nonroot
WORKDIR /app
COPY --from=build /out/api /app/api
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/app/api"]
