from logging import getLogger
from typing import Any

from datasets import Dataset  # type: ignore[import-untyped]
from pandas import DataFrame, read_csv

from constants import SYSTEM_PROMPT
from dataset.constants import (
    DATASET_PATH,
    DATASET_SLICE_BY_SPLIT_TYPE,
    TRAINING_PROMPT,
    WNC_COLUMNS,
)
from dataset.enums import DatasetSplit, TokenizationType, WNCColumn

logger = getLogger(__name__)


def get_dataset_split(
    dataset_split: DatasetSplit, tokenization_type: TokenizationType, tokenizer: Any
) -> Dataset:
    dataset = get_preprocessed_dataset_slice(DATASET_SLICE_BY_SPLIT_TYPE[dataset_split])
    tokenized_dataset = TOKENIZE_FUNCTION_BY_TYPE[tokenization_type](dataset, tokenizer)

    logger.info(
        f"Loaded {len(tokenized_dataset)} examples for {dataset_split} split and {tokenization_type} task"
    )

    return tokenized_dataset


def get_preprocessed_dataset_slice(slice: slice) -> Dataset:
    dataset = load_wnc_from_csv()
    sliced_dataset = dataset.iloc[slice]

    preprocessed_dataframe = preprocess_data(sliced_dataset)
    processed_dataset = Dataset.from_pandas(preprocessed_dataframe)

    return processed_dataset


def load_wnc_from_csv() -> DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No file named 'biased.full' found at {DATASET_PATH}, have you ran download-dataset?"
        )
    logger.info(f"Loading dataset from {DATASET_PATH}")

    return read_csv(
        DATASET_PATH, delimiter="\t", on_bad_lines="warn", names=WNC_COLUMNS
    )


def preprocess_data(dataframe: DataFrame) -> DataFrame:
    reduced_dataframe = dataframe.drop(
        columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags", "src_POS_tags"]
    )
    reduced_dataframe = reduced_dataframe.rename(
        columns={"src_raw": WNCColumn.BIASED, "tgt_raw": WNCColumn.NEUTRAL}
    )

    return reduced_dataframe.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL])


def tokenize_data(batch: dict, tokenizer: Any) -> dict:
    biased_texts = batch[WNCColumn.BIASED]
    neutral_texts = batch[WNCColumn.NEUTRAL]

    messages = [
        [
            {"role": "user", "content": f"{TRAINING_PROMPT}:\nBiased text: {biased}"},
            {"role": "assistant", "content": neutral},
        ]
        for biased, neutral in zip(biased_texts, neutral_texts)
    ]

    full_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        max_length=512,
        padding="max_length",
        truncation=True,
        enable_thinking=False,
        return_dict=True,
    )

    user_tokens = tokenizer.apply_chat_template(
        [[msg[0]] for msg in messages],
        tokenize=True,
        add_generation_prompt=True,
        max_length=512,
        truncation=True,
        enable_thinking=False,
    )

    labels = []
    for i in range(len(full_tokens["input_ids"])):
        user_len = sum(1 for t in user_tokens[i] if t != tokenizer.pad_token_id)
        label = [-100] * user_len + full_tokens["input_ids"][i][user_len:]
        labels.append(label)

    return {
        "input_ids": full_tokens["input_ids"],
        "labels": labels,
        "attention_mask": full_tokens["attention_mask"],
        "biased": batch[WNCColumn.BIASED],
        "neutral": batch[WNCColumn.NEUTRAL],
    }


def map_grpo_data(sample: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Make this neutral: {sample[WNCColumn.BIASED]}",
            },
            {"role": "assistant", "content": " "},
        ],
    }


def map_dpo_data(sample: dict, tokenizer: Any) -> dict:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Make this neutral: {sample[WNCColumn.BIASED]}",
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    return {
        "prompt": prompt_messages,
        "formatted_prompt": formatted_prompt,
        WNCColumn.BIASED: sample[WNCColumn.BIASED],
        WNCColumn.NEUTRAL: sample[WNCColumn.NEUTRAL],
    }


def tokenize_for_sft(dataset: Dataset, tokenizer: Any) -> Dataset:
    return dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )


def tokenize_for_grpo(dataset: Dataset, _) -> Dataset:
    return dataset.map(
        lambda data: map_grpo_data(data),
        batched=False,
    )


def tokenize_for_dpo(dataset: Dataset, tokenizer: Any) -> Dataset:
    return dataset.map(
        lambda data: map_dpo_data(data, tokenizer),
        batched=False,
    )


def tokenize_for_distillation(dataset: Dataset, tokenizer: Any) -> Dataset:
    return dataset.map(
        lambda data: tokenize_data(data, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    ).remove_columns([WNCColumn.BIASED, WNCColumn.NEUTRAL])


TOKENIZE_FUNCTION_BY_TYPE = {
    TokenizationType.SFT: tokenize_for_sft,
    TokenizationType.GRPO: tokenize_for_grpo,
    TokenizationType.DPO: tokenize_for_dpo,
    TokenizationType.DISTIL: tokenize_for_distillation,
}
