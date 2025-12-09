from pathlib import Path

from dataset.enums import DatasetSplit

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATASET_NAME = "chandiragunatilleke/wiki-neutrality-corpus"
DATASET_DOWNLOAD_PATH = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DOWNLOAD_PATH / "WNC" / "WNC" / "biased.full"

WNC_COLUMNS = [
    "id",
    "src_tok",
    "tgt_tok",
    "src_raw",
    "tgt_raw",
    "src_POS_tags",
    "tgt_parse_tags",
]
TRAINING_PROMPT = "You are an expert at text style transfer. Your task is to transform text from biased to neutral."
INSTRUCTION_PROMPT = "You are an expert at text style transfer. Your task is to transform text from biased to neutral."

DATASET_SLICE_BY_SPLIT_TYPE = {
    DatasetSplit.TRAIN: slice(0, 3000),
    DatasetSplit.VALIDATION: slice(3000, 3500),
    DatasetSplit.TEST: slice(-300, None),
}
