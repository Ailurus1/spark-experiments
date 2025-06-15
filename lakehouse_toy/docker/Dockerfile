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
RUN mkdir -p data

RUN uv pip install --system -r pyproject.toml

ENV PYTHONPATH=/app
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN mkdir -p /tmp/spark-temp

RUN echo '#!/bin/bash\n\
if [ -d "data" ] && [ "$(ls -A data)" ]; then\n\
    echo "Data directory exists and is not empty, skipping download"\n\
else\n\
    echo "Downloading dataset..."\n\
    python3 src/download_dataset.py\n\
fi\n\
\n\
echo "Starting application..."\n\
python3 src/app.py\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
