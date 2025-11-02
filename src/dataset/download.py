import os
import shutil
from logging import getLogger

from kagglehub import dataset_download  # type: ignore[import-untyped]

from dataset.constants import DATASET_DOWNLOAD_PATH

logger = getLogger(__name__)


def download_wnc() -> None:
    path = dataset_download(
        "chandiragunatilleke/wiki-neutrality-corpus", force_download=True
    )
    logger.info(f"Downloaded dataset to {path}")

    try:
        if os.access(path, os.W_OK):
            shutil.move(path, DATASET_DOWNLOAD_PATH)
        else:
            logger.info("Source directory is read-only; copying instead of moving...")
            shutil.copytree(path, DATASET_DOWNLOAD_PATH, dirs_exist_ok=True)

        wnc_subdir = DATASET_DOWNLOAD_PATH / "1" / "WNC"
        if wnc_subdir.exists():
            shutil.move(wnc_subdir, DATASET_DOWNLOAD_PATH)
            shutil.rmtree(DATASET_DOWNLOAD_PATH / "1", ignore_errors=True)

    except Exception as e:
        logger.error(f"Error moving dataset to {DATASET_DOWNLOAD_PATH}: {e}")
        raise

    logger.info(f"Moved dataset to {DATASET_DOWNLOAD_PATH}")
