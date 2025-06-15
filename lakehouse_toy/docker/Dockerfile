FROM ghcr.io/astral-sh/uv:0.5.5-python3.12-alpine

WORKDIR /app

RUN apk update && \
    apk upgrade && \
    apk add --no-cache \
    gcc \
    musl-dev \
    openjdk17-jdk \
    curl \
    build-base \
    linux-headers

COPY pyproject.toml* ./
COPY src/ ./src/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN mkdir -p data

RUN uv pip install --system -r pyproject.toml

ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN mkdir -p /tmp/spark-temp

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
