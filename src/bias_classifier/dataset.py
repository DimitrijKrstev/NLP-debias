from logging import getLogger

from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

from bias_classifier.constants import LABEL_BIASED, LABEL_NEUTRAL, MAX_LENGTH
from dataset.constants import DATASET_SLICE_BY_SPLIT_TYPE
from dataset.enums import DatasetSplit, WNCColumn
from dataset.preprocess import get_preprocessed_dataset_slice

logger = getLogger(__name__)


def get_classification_dataset(
    split: DatasetSplit,
    tokenizer: PreTrainedTokenizer,
    max_samples: int | None = None,
) -> Dataset:
    slice_range = DATASET_SLICE_BY_SPLIT_TYPE[split]
    raw_dataset = get_preprocessed_dataset_slice(slice_range)

    if max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    classification_dataset = transform_to_classification(raw_dataset)
    tokenized_dataset = tokenize_dataset(classification_dataset, tokenizer)

    logger.info(
        f"Created classification dataset for {split}: {len(tokenized_dataset)} samples "
        f"(from {len(raw_dataset)} WNC pairs)"
    )

    return tokenized_dataset


def transform_to_classification(dataset: Dataset) -> Dataset:
    biased_samples = dataset.map(
        lambda x: {
            "text": x[WNCColumn.BIASED],
            "label": LABEL_BIASED,
        },
        remove_columns=dataset.column_names,
    )

    neutral_samples = dataset.map(
        lambda x: {
            "text": x[WNCColumn.NEUTRAL],
            "label": LABEL_NEUTRAL,
        },
        remove_columns=dataset.column_names,
    )

    combined = concatenate_datasets([biased_samples, neutral_samples])
    shuffled = combined.shuffle(seed=42)

    logger.info(
        f"Transformed WNC pairs to classification: "
        f"{len(biased_samples)} biased + {len(neutral_samples)} neutral = {len(combined)} total"
    )

    return shuffled


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
    )

    return tokenized
