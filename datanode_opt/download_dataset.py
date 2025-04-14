import kagglehub
from pathlib import Path
import pandas as pd


def download_anime_dataset():
    path = Path(kagglehub.dataset_download("dbdmobile/myanimelist-dataset"))
    source = path / "final_animedataset.csv"
    destination = Path("data")
    destination.mkdir(exist_ok=True)

    target = destination / "anime_dataset.parquet"
    if not target.exists():
        df = pd.read_csv(source)
        df.to_parquet(target)
        print(f"Dataset downloaded and converted to parquet at {target}")
    else:
        print(f"Dataset already exists at {target}")


if __name__ == "__main__":
    download_anime_dataset()
