import kagglehub
from pathlib import Path


def download_dataset():
    destination = Path("data")
    destination.mkdir(exist_ok=True)
    target = destination / "data.csv"

    if not target.exists():
        path = Path(
            kagglehub.dataset_download("parisrohan/credit-score-classification")
        )
        source = path / "train.csv"
        target.write_bytes(source.read_bytes())
        print(f"Dataset downloaded to {target}")
    else:
        print(f"Dataset already exists at {target}")


if __name__ == "__main__":
    download_dataset()
