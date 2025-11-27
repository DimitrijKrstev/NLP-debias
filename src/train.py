import logging
from os import environ

import mlflow
from transformers import DataCollatorForLanguageModeling, Trainer

from constants import TRAIN_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from utils import get_training_args, load_peft_model, load_tokenizer

environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def train_model(
    quantize: bool,
    mlflow_experiment: str,
    model_name: str,
) -> None:
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=f"{model_name}-debiasing")

    model = load_peft_model(model_name, quantize)
    tokenizer = load_tokenizer(model_name)
    model.train()

    logger.info("Loading and pre-processing dataset")
    train_dataset = get_dataset_split(
        DatasetSplit.TRAIN, TokenizationType.SFT, tokenizer
    )
    val_dataset = get_dataset_split(
        DatasetSplit.VALIDATION, TokenizationType.SFT, tokenizer
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model and tokenizer to {TRAIN_OUTPUT_DIR}")
    trainer.save_model(TRAIN_OUTPUT_DIR)
    tokenizer.save_pretrained(TRAIN_OUTPUT_DIR)

    logger.info("Training complete!")
    mlflow.end_run()
