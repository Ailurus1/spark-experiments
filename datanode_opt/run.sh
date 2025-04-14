#!/bin/bash

mkdir -p data

echo "Downloading dataset..."
python download_dataset.py

if [ $? -eq 0 ]; then
    python app.py "$@"
else
    echo "Dataset download failed!"
    exit 1
fi
