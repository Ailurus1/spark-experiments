FROM ghcr.io/astral-sh/uv:0.5.5-python3.12-bookworm

WORKDIR /app

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    gcc \
    musl-dev \
    openjdk-17-jdk \
    curl

COPY pyproject.toml* ./
COPY src/ ./src/
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN mkdir -p data

RUN uv pip install --system -r pyproject.toml

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
