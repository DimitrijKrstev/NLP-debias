from logging import getLogger
from typing import Any

from datasets import Dataset
from pandas import DataFrame, read_csv

from dataset.constants import DATASET_PATH, TRAINING_PROMPT, WNC_COLUMNS, WNCColumn

logger = getLogger(__name__)


# TODO make generic
def get_train_dataset(tokenizer: Any) -> Dataset:
    dataset = load_wnc_from_csv()
    dataset = dataset[-3000:]

    train_idx = int(0.8 * len(dataset))

    train_set = dataset[:train_idx].copy()
    preprocessed_train_set = preprocess_data(train_set)
    train_dataset = Dataset.from_pandas(preprocessed_train_set)

    tokenized_train_dataset = train_dataset.map(
        lambda data: tokenize_data(data, tokenizer), batched=True
    )

    logger.info(f"Loaded {len(train_dataset)} training examples")

    return tokenized_train_dataset


def get_test_dataset(tokenizer: Any) -> Dataset:
    dataset = load_wnc_from_csv()
    dataset = dataset[-1000:]

    preprocessed_dataframe = preprocess_data(dataset)

    test_dataset = Dataset.from_pandas(preprocessed_dataframe)

    test_dataset = test_dataset.map(
        lambda data: tokenize_data(data, tokenizer), batched=True
    )

    logger.info(f"Loaded {len(test_dataset)} test examples")

    return test_dataset


def get_train_val_split(tokenizer: Any) -> tuple[Dataset, Dataset]:
    dataset = load_wnc_from_csv()
    dataset = dataset[:3000]

    train_idx = int(0.8 * len(dataset))
    val_idx = int(0.9 * len(dataset))

    train_set = dataset[:train_idx].copy()
    val_set = dataset[train_idx:val_idx].copy()

    preprocessed_train_set = preprocess_data(train_set)
    preprocessed_val_set = preprocess_data(val_set)

    train_dataset = Dataset.from_pandas(preprocessed_train_set)
    val_dataset = Dataset.from_pandas(preprocessed_val_set)

    tokenized_train_dataset = train_dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val_dataset = val_dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
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


def tokenize_data(batch: dict, tokenizer: Any) -> dict:
    biased_texts = batch[WNCColumn.BIASED]
    neutral_texts = batch[WNCColumn.NEUTRAL]

    all_input_ids = []
    all_labels = []
    all_attention_masks = []

    for biased_text, neutral_text in zip(biased_texts, neutral_texts):
        messages = [
            {
                "role": "user",
                "content": f"{TRAINING_PROMPT}:\nBiased text: {biased_text}",
            },
            {"role": "assistant", "content": neutral_text},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
            max_length=256,
            padding="max_length",
            truncation=True,
        )

        user_only = tokenizer.apply_chat_template(
            [messages[0]],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            max_length=256,
            padding="max_length",
            truncation=True,
        )

        labels = input_ids.copy()
        labels[: len(user_only)] = [-100] * len(user_only)

        attention_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    }


# def tokenize_test_data(batch: dict, tokenizer: Any) -> dict:
#     biased_text = batch[WNCColumn.BIASED]
#     neutral_text = batch[WNCColumn.NEUTRAL]

#     tokenized = tokenizer(
#         biased_text,
#         truncation=True,
#         max_length=256,
#         padding=False,
#         return_attention_mask=True,
#     )

#     tokenized["reference_neutral"] = neutral_text

#     return tokenized
