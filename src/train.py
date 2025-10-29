import logging
from os import environ

import mlflow
from transformers import DataCollatorForLanguageModeling, Trainer

from constants import TRAIN_OUTPUT_DIR
from dataset.preprocess import get_train_val_split
from utils import get_training_args, load_peft_model_and_tokenizer

environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def train_model(
    quantize: bool,
    mlflow_experiment: str,
    model_name: str,
) -> None:
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=f"{model_name}-debiasing")

    model, tokenizer = load_peft_model_and_tokenizer(model_name, quantize)
    model.train()

    logger.info("Loading and pre-processing dataset")
    train_dataset, val_dataset = get_train_val_split(tokenizer)
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
