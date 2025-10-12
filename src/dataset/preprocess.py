from logging import getLogger
from typing import Any

from datasets import Dataset
from pandas import DataFrame, read_csv

from dataset.constants import DATASET_PATH, WNC_COLUMNS, WNCColumn

logger = getLogger(__name__)


def get_train_test_dataset(tokenizer: Any) -> tuple[Dataset, Dataset]:
    dataframe = load_wnc_from_csv()
    dataframe = get_sft_dataset_portion(dataframe)
    preprocessed_dataframe = preprocess_data(dataframe)
    train_dataset, test_dataset = split_and_tokenize_rows(
        preprocessed_dataframe, tokenizer
    )

    logger.info(
        f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples"
    )

    return train_dataset, test_dataset


def load_wnc_from_csv() -> DataFrame:
    dataset_path = DATASET_PATH

    main_file = dataset_path / "biased.full"
    if not main_file.exists():
        raise FileNotFoundError(
            f"No file named 'biased.full' found at {main_file}, have you ran download-dataset?"
        )
    logger.info(f"Loading dataset from {main_file}")

    return read_csv(main_file, delimiter="\t", on_bad_lines="warn", names=WNC_COLUMNS)


def get_sft_dataset_portion(dataframe: DataFrame) -> DataFrame:
    return dataframe[: int(0.85 * len(dataframe))]


def preprocess_data(dataframe: DataFrame) -> DataFrame:
    dataframe.drop(columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags"], inplace=True)
    dataframe = dataframe.rename(
        columns={"src_raw": WNCColumn.BIASED, "tgt_raw": WNCColumn.NEUTRAL}
    )

    dataframe = dataframe.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL])

    return dataframe


def split_and_tokenize_rows(
    dataframe: DataFrame, tokenizer: Any
) -> tuple[Dataset, Dataset]:
    split_idx = int(0.9 * len(dataframe))
    train_set = dataframe[:split_idx]
    test_set = dataframe[split_idx:]

    train_dataset = Dataset.from_pandas(train_set)
    test_dataset = Dataset.from_pandas(test_set)

    train_dataset = train_dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    return train_dataset, test_dataset


def tokenize_data(batch: dict, tokenizer: Any) -> dict:
    biased_texts = batch[WNCColumn.BIASED]
    neutral_texts = batch[WNCColumn.NEUTRAL]

    concatenated_texts = [f"{b}\n{n}" for b, n in zip(biased_texts, neutral_texts)]

    tokenized = tokenizer(
        concatenated_texts,
        truncation=True,
        max_length=512,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized
