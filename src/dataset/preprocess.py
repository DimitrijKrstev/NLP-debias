from logging import getLogger
from typing import Any

from datasets import Dataset
from pandas import DataFrame, read_csv

from dataset.constants import DATASET_PATH, TRAINING_PROMPT, WNC_COLUMNS, WNCColumn

logger = getLogger(__name__)


def get_dataset_slice(
    tokenizer: Any,
    start_idx: int | None = None,
    end_idx: int | None = None,
    max_samples: int | None = None,
    remove_columns: bool = False,
) -> Dataset:

    dataset = load_wnc_from_csv()

    if max_samples is not None:
        dataset = dataset.iloc[:max_samples]

    if start_idx is not None or end_idx is not None:
        dataset = dataset.iloc[start_idx:end_idx]

    preprocessed_dataframe = preprocess_data(dataset)
    processed_dataset = Dataset.from_pandas(preprocessed_dataframe)

    tokenize_fn = lambda data: tokenize_data(data, tokenizer)

    tokenized_dataset = processed_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=processed_dataset.column_names if remove_columns else None,
    )

    logger.info(f"Loaded {len(processed_dataset)} examples")

    return tokenized_dataset


def get_train_dataset(tokenizer: Any) -> Dataset:
    train_idx = int(0.8 * 3000)

    return get_dataset_slice(tokenizer, start_idx=0, end_idx=train_idx)


def get_test_dataset(tokenizer: Any) -> Dataset:
    return get_dataset_slice(tokenizer, start_idx=-1000, end_idx=None)


def get_train_val_split(tokenizer: Any) -> tuple[Dataset, Dataset]:
    train_idx = int(0.8 * 3000)
    val_idx = int(0.9 * 3000)

    train_dataset = get_dataset_slice(
        tokenizer, start_idx=0, end_idx=train_idx, max_samples=3000, remove_columns=True
    )
    val_dataset = get_dataset_slice(
        tokenizer,
        start_idx=train_idx,
        end_idx=val_idx,
        max_samples=3000,
        remove_columns=True,
    )

    logger.info(
        f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples"
    )

    return train_dataset, val_dataset


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


def tokenize_data(batch: dict, tokenizer: Any) -> dict:
    biased_texts = batch[WNCColumn.BIASED]
    neutral_texts = batch[WNCColumn.NEUTRAL]

    messages = [
        [
            {"role": "user", "content": f"{TRAINING_PROMPT}:\nBiased text: {b}"},
            {"role": "assistant", "content": n},
        ]
        for b, n in zip(biased_texts, neutral_texts)
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    user_only = tokenizer.apply_chat_template(
        [
            [{"role": "user", "content": f"{TRAINING_PROMPT}:\nBiased text: {b}"}]
            for b in biased_texts
        ],
        tokenize=True,
        add_generation_prompt=True,
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    labels = []
    for i in range(len(input_ids)):
        label = input_ids[i].copy()
        user_length = sum(
            1 for token in user_only[i] if token != tokenizer.pad_token_id
        )
        label[:user_length] = [-100] * user_length
        labels.append(label)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [[1] * len(ids) for ids in input_ids],
        "biased_text": batch[WNCColumn.BIASED],
        "neutral_text": batch[WNCColumn.NEUTRAL],
    }
