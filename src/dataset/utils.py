from logging import getLogger

from datasets import Dataset
from pandas import DataFrame, read_csv

from src.dataset.constants import DATASET_PATH, WNCColumn
from src.models import WNCData

logger = getLogger(__name__)


def get_train_test_dataset() -> tuple[Dataset, Dataset]:
    dataframe = load_wnc_from_csv()
    rows = preprocess_data(dataframe)
    train_dataset, test_dataset = split_and_tokenize_rows(rows)

    return train_dataset, test_dataset


def load_wnc_from_csv() -> DataFrame:
    dataset_path = DATASET_PATH

    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {dataset_path}, have you ran download-dataset?"
        )

    main_file = next(
        (file for file in csv_files if file.name.lower() == "biased.full"), None
    )
    if not main_file:
        main_file = csv_files[0]
        logger.warning(
            "No file named 'biased.full' found, have you ran download-dataset?"
        )
    logger.info(f"Loading dataset from {main_file}")

    return read_csv(main_file)


def preprocess_data(dataframe: DataFrame) -> list[WNCData]:
    dataframe.drop(columns=["id", "src_tok", "tgt_tok", "tgt_parse_tags"], inplace=True)
    dataframe = dataframe.rename(
        columns={"src_raw": WNCColumn.BIASED, "tgt_raw": WNCColumn.NEUTRAL}
    )

    dataframe = dataframe.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL])

    rows = [
        WNCData(
            (
                "Rewrite the following text to remove bias while preserving the core information:\n\n"
                + f"Text: {row[WNCColumn.BIASED]}\n\n"
                + f"Rewritten: {row[WNCColumn.NEUTRAL]}"
            ),
            row[WNCColumn.BIASED],
            row[WNCColumn.NEUTRAL],
        )
        for _, row in dataframe.iterrows()
    ]

    return rows


def split_and_tokenize_rows(data: list[WNCData]) -> tuple[Dataset, Dataset]:
    split_idx = int(0.9 * len(data))
    train_set = data[:split_idx]
    test_set = data[split_idx:]

    train_dataset = Dataset.from_list([data.to_dict() for data in train_set])
    test_dataset = Dataset.from_list([data.to_dict() for data in test_set])

    train_dataset = train_dataset.map(tokenize_data, batched=True)
    test_dataset = test_dataset.map(tokenize_data, batched=True)

    logger.info(
        f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples"
    )

    return train_dataset, test_dataset


def tokenize_data(data: dict, tokenizer) -> dict:
    model_inputs = tokenizer(
        data[WNCColumn.BIASED],
        truncation=True,
        # Skip padding, data collator does batched padding
        padding=False,
    )

    labels = tokenizer(
        data[WNCColumn.NEUTRAL],
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def create_debiasing_prompt(biased_text: str) -> str:
    return (
        "Rewrite the following text to remove bias while preserving the core information:\n\n"
        + f"Text: {biased_text}"
        + "Rewritten:\n"
    )
