import shutil
from logging import getLogger

from kagglehub import dataset_download

from dataset.constants import DATASET_DOWNLOAD_PATH

logger = getLogger(__name__)


def download_wnc() -> None:
    path = dataset_download(
        "chandiragunatilleke/wiki-neutrality-corpus", force_download=True
    )
    logger.info(f"Downloaded dataset to {path}")

    try:
        shutil.move(path, DATASET_DOWNLOAD_PATH)
        shutil.move(DATASET_DOWNLOAD_PATH / "1" / "WNC", DATASET_DOWNLOAD_PATH)
        shutil.rmtree(DATASET_DOWNLOAD_PATH / "1")
    except Exception as e:
        logger.error(f"Error moving dataset to {DATASET_DOWNLOAD_PATH}: {e}")
        raise

    logger.info(f"Moved dataset to {DATASET_DOWNLOAD_PATH}")
