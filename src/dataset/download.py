import shutil
from logging import getLogger

from kagglehub import dataset_download

from dataset.constants import DATASET_PATH

logger = getLogger(__name__)


def download_wnc() -> None:
    path = dataset_download(
        "chandiragunatilleke/wiki-neutrality-corpus", force_download=True
    )
    logger.info(f"Downloaded dataset to {path}")

    new_directory = DATASET_PATH
    try:
        shutil.move(path, new_directory)
    except Exception as e:
        logger.error(f"Error moving dataset to {new_directory}: {e}")
        raise

    logger.info(f"Moved dataset to {new_directory}")
