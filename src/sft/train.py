import logging

import mlflow
from transformers import DataCollatorForLanguageModeling, Trainer

from constants import TRAIN_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from sft.utils import get_training_args
from utils import load_unsloth_model

logger = logging.getLogger(__name__)


def train_model(quantize: bool, mlflow_experiment: str, model_name: str) -> None:
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"{model_name}-SFT"):
        training_args = get_training_args()

        mlflow.set_tag("task", "training")
        mlflow.set_tag("training_type", "sft")

        model, tokenizer = load_unsloth_model(model_name, quantize)
        model.train()

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
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Saving model and tokenizer to {TRAIN_OUTPUT_DIR}")
        trainer.save_model(TRAIN_OUTPUT_DIR)

    logger.info("Training complete!")
