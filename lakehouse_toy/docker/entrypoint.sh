#!/bin/sh

if [ -d "data" ] && [ "$(ls -A data)" ]; then
    echo "Data directory exists and is not empty, skipping download"
else
    echo "Downloading dataset..."
    python3 src/download_dataset.py
fi

echo "Starting application..."
python3 src/app.py