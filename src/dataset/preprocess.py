from logging import getLogger
from typing import Any

from datasets import Dataset
from pandas import DataFrame, read_csv

from dataset.constants import DATASET_PATH, WNC_COLUMNS, WNCColumn

logger = getLogger(__name__)


def get_test_dataset(tokenizer: Any) -> Dataset:
    dataset = load_wnc_from_csv()
    dataset = dataset[:-3000]

    test_idx = int(0.9 * len(dataset))
    data = dataset[test_idx:].copy()
    preprocessed_dataframe = preprocess_data(data)

    test_dataset = Dataset.from_pandas(preprocessed_dataframe)

    test_dataset = test_dataset.map(
        lambda data: tokenize_test_data(data, tokenizer),
    )

    logger.info(f"Loaded {len(test_dataset)} test examples")

    return test_dataset


def get_train_val_split(tokenizer: Any, is_sft: bool) -> tuple[Dataset, Dataset]:
    dataset = load_wnc_from_csv()
    dataset = dataset[:-3000] if is_sft else dataset[-3000:]

    train_idx = int(0.8 * len(dataset))
    val_idx = int(0.9 * len(dataset))

    train_set = dataset[:train_idx].copy()
    val_set = dataset[train_idx:val_idx].copy()

    preprocessed_train_set = preprocess_data(train_set)
    preprocessed_val_set = preprocess_data(val_set)

    train_dataset = Dataset.from_pandas(preprocessed_train_set)
    val_dataset = Dataset.from_pandas(preprocessed_val_set)

    tokenized_train_dataset = train_dataset.map(
        lambda data: tokenize_train_data(data, tokenizer)
    )
    tokenized_val_dataset = val_dataset.map(
        lambda data: tokenize_train_data(data, tokenizer)
    )

    logger.info(
        f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples"
    )

    return tokenized_train_dataset, tokenized_val_dataset


def load_wnc_from_csv() -> DataFrame:
    dataset_path = DATASET_PATH

    main_file = dataset_path / "biased.full"
    if not main_file.exists():
        raise FileNotFoundError(
            f"No file named 'biased.full' found at {main_file}, have you ran download-dataset?"
        )
    logger.info(f"Loading dataset from {main_file}")

    return read_csv(main_file, delimiter="\t", on_bad_lines="warn", names=WNC_COLUMNS)


def preprocess_data(dataframe: DataFrame) -> DataFrame:
    dataframe.drop(columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags"], inplace=True)
    dataframe = dataframe.rename(
        columns={"src_raw": WNCColumn.BIASED, "tgt_raw": WNCColumn.NEUTRAL}
    )

    dataframe = dataframe.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL])

    return dataframe


def tokenize_train_data(batch: dict, tokenizer: Any) -> dict:
    biased_text = batch[WNCColumn.BIASED]
    neutral_text = batch[WNCColumn.NEUTRAL]

    concatenated_text = f"{biased_text}\n{neutral_text}"

    tokenized = tokenizer(
        concatenated_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
    )

    # DataCollatorForCausalLM will handle labels internally
    # tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def tokenize_test_data(batch: dict, tokenizer: Any) -> dict:
    biased_text = batch[WNCColumn.BIASED]
    neutral_text = batch[WNCColumn.NEUTRAL]

    tokenized = tokenizer(
        biased_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_attention_mask=True,
    )

    tokenized["reference_neutral"] = neutral_text

    return tokenized
