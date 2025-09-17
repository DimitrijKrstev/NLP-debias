from enum import StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "data"


class WNCColumn(StrEnum):
    BIASED = "biased"
    NEUTRAL = "neutral"
