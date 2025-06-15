# Lakehouse Toy project

Dataset: https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## How to run

### Option 1: Docker
```bash
docker compose up
```
Wait until services are up. Then you can open MLFlow (http://0.0.0.0:5000) and model + metrics (F1, AUC-ROC, Average Precision) will be logged there.

### Option 2: Standalone
```bash
uv pip install -r pyproject.toml

python3 src/download_dataset.py # will put a .csv file into `data/`
python3 src/app.py
```
