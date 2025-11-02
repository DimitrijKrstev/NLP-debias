from logging import getLogger
from typing import Any

from datasets import Dataset  # type: ignore[import-untyped]
from pandas import DataFrame, read_csv

from constants import GRPO_SYSTEM_PROMPT
from dataset.constants import DATASET_PATH, TRAINING_PROMPT, WNC_COLUMNS, WNCColumn

logger = getLogger(__name__)


def get_dataset_slice(
    tokenizer: Any,
    start_idx: int = 0,
    end_idx: int | None = None,
    is_rl: bool = False,
) -> Dataset:
    dataset = load_wnc_from_csv()

    if end_idx is not None:
        dataset = dataset.iloc[start_idx:end_idx]
    else:
        dataset = dataset.iloc[start_idx:]

    preprocessed_dataframe = preprocess_data(dataset)
    processed_dataset = Dataset.from_pandas(preprocessed_dataframe)

    if is_rl:
        tokenized_dataset = processed_dataset.map(
            lambda data: map_grpo_data(data),
            batched=False,
        )
    else:
        tokenized_dataset = processed_dataset.map(
            lambda data: tokenize_data(data, tokenizer),
            batched=True,
            remove_columns=processed_dataset.column_names,
        )

    logger.info(f"Loaded {len(processed_dataset)} examples")

    return tokenized_dataset


def get_train_dataset(tokenizer: Any, is_rl: bool = False) -> Dataset:
    return get_dataset_slice(tokenizer, end_idx=3000, is_rl=is_rl)


def get_test_dataset(tokenizer: Any, is_rl: bool = False) -> Dataset:
    return get_dataset_slice(tokenizer, start_idx=-1000, end_idx=None, is_rl=is_rl)


def get_train_val_split(tokenizer: Any) -> tuple[Dataset, Dataset]:
    train_val_sep_idx = int(0.9 * 3000)

    train_dataset = get_dataset_slice(tokenizer, end_idx=train_val_sep_idx)
    val_dataset = get_dataset_slice(
        tokenizer, start_idx=train_val_sep_idx, end_idx=3000
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
    dataframe.drop(
        columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags", "src_POS_tags"],
        inplace=True,
    )
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
        [[message[0]] for message in messages],
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


def map_grpo_data(sample: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": sample[WNCColumn.BIASED]},
        ],
    }
