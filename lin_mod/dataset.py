import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
import typer

from lin_mod.config import RAW_DATA_DIR
from kaggle.api.kaggle_api_extended import KaggleApi

# Load .env and Kaggle credentials
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()

app = typer.Typer()

@app.command()
def main(
        dataset_ref: str = typer.Option(..., help="Kaggle dataset reference"),
        path: str = typer.Option(None, help="Download path")
):
    """Download raw insurance data, clean it, and save it to a local CSV file."""
    download_path = Path(path) if path else RAW_DATA_DIR
    download_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset {dataset_ref} to {download_path}...")
    api.dataset_download_files(dataset_ref, path=str(download_path), unzip=True)
    logger.success("Download complete.")

if __name__ == "__main__":
    app()
