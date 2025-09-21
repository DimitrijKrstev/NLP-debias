from logging import getLogger
from typing import Any

from datasets import Dataset
from pandas import DataFrame, read_csv

from dataset.constants import DATASET_PATH, WNC_COLUMNS, WNCColumn
from models import WNCData

logger = getLogger(__name__)


def get_train_test_dataset(tokenizer: Any) -> tuple[Dataset, Dataset]:
    dataframe = load_wnc_from_csv()
    rows = preprocess_data(dataframe)
    train_dataset, test_dataset = split_and_tokenize_rows(rows, tokenizer)

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


def preprocess_data(dataframe: DataFrame) -> list[WNCData]:
    dataframe.drop(columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags"], inplace=True)
    dataframe = dataframe.rename(
        columns={"src_raw": WNCColumn.BIASED, "tgt_raw": WNCColumn.NEUTRAL}
    )

    dataframe = dataframe.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL])

    rows = [
        WNCData(
            (
                "Rewrite the following text to remove bias "
                + "while preserving the core information:\n\n"
                + f"Text: {row[WNCColumn.BIASED]}\n\n"
                + f"Rewritten: {row[WNCColumn.NEUTRAL]}"
            ),
            row[WNCColumn.BIASED],
            row[WNCColumn.NEUTRAL],
        )
        for _, row in dataframe.iterrows()
    ]

    return rows


def split_and_tokenize_rows(
    data: list[WNCData], tokenizer: Any
) -> tuple[Dataset, Dataset]:
    split_idx = int(0.9 * len(data))
    train_set = data[:split_idx]
    test_set = data[split_idx:]

    train_dataset = Dataset.from_list([data.to_dict() for data in train_set])
    test_dataset = Dataset.from_list([data.to_dict() for data in test_set])

    train_dataset = train_dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    return train_dataset, test_dataset


def tokenize_data(data: dict, tokenizer: Any) -> dict:
    batched_inputs = data["model_input_text"]
    batched_targets = data[WNCColumn.NEUTRAL]

    results = {
        "input_ids": [],
        "attention_mask": [],
    }

    full_texts = [
        f"{input_text}\n\nOutput:\n{target_text}{tokenizer.eos_token}"
        for input_text, target_text in zip(batched_inputs, batched_targets)
    ]

    tokens_list = [
        tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None,
        )
        for text in full_texts
    ]
    results["input_ids"] = [tokens["input_ids"] for tokens in tokens_list]
    results["attention_mask"] = [tokens["attention_mask"] for tokens in tokens_list]

    return results


def create_debiasing_prompt(biased_text: str) -> str:
    return (
        "Rewrite the following text to remove bias while preserving the core information:\n\n"
        + f"Text: {biased_text}"
        + "Rewritten:\n"
    )
