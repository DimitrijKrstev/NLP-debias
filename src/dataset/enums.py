from enum import StrEnum


class WNCColumn(StrEnum):
    BIASED = "biased"
    NEUTRAL = "neutral"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TokenizationType(StrEnum):
    SFT = "SFT"
    EVAL = "EVAL"
    GRPO = "GRPO"
    DISTIL = "DISTIL"
    DPO = "DPO"
