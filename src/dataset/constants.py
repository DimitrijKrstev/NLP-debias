from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATASET_DOWNLOAD_PATH = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DOWNLOAD_PATH / "WNC" / "WNC"

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


class WNCColumn(StrEnum):
    BIASED = "biased"
    NEUTRAL = "neutral"
