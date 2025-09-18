from pathlib import Path
import os

import typer
from loguru import logger

from .config import RAW_DATA_DIR

app = typer.Typer(help="CLI for managing datasets.")

def setup_kaggle():
    """Load .env and authenticate Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from dotenv import load_dotenv

    dotenv_path = Path(__file__).parent / ".env"
    if dotenv_path.exists(): load_dotenv(dotenv_path)

    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

    api = KaggleApi()
    api.authenticate()
    return api

@app.command()
def download(
        dataset_ref: str = typer.Option("mirichoi0218/insurance", help="Path to insurance dataset."),
        path: Path = typer.Option(RAW_DATA_DIR, help="Raw data path"),
):
    """
    Download a dataset from Kaggle and save it locally in CSV.
    - dataset_ref: Kaggle dataset reference (username/dataset-name)
    - path: local path to save data (defaults to RAW_DATA_DIR)
    """
    api = setup_kaggle()
    download_path = Path(path)
    download_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset {dataset_ref} to {download_path}...")
    api.dataset_download_files(dataset_ref, path=str(download_path), unzip=True)
    logger.success("Download complete.")

if __name__ == "__main__":
    app()