import kagglehub
from pathlib import Path


def download_anime_dataset():
    path = Path(kagglehub.dataset_download("dbdmobile/myanimelist-dataset"))
    source = path / "final_animedataset.csv"
    destination = Path("data")
    destination.mkdir(exist_ok=True)

    target = destination / "final_animedataset.csv"
    if not target.exists():
        target.write_bytes(source.read_bytes())
        print(f"Dataset downloaded to {target}")
    else:
        print(f"Dataset already exists at {target}")


if __name__ == "__main__":
    download_anime_dataset()
