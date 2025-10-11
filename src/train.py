import logging

import mlflow
from transformers import DataCollatorForLanguageModeling, Trainer

from constants import OUTPUT_DIR
from dataset.preprocess import get_train_test_dataset
from utils import get_training_args, load_peft_model_and_tokenizer

logger = logging.getLogger(__name__)


def train_model(
    quantize: bool,
    mlflow_experiment: str,
    model_name: str,
):
    mlflow.set_experiment(mlflow_experiment)
    # TODO sekoj run treba da e poseben
    mlflow.start_run(run_name=f"{model_name}-debiasing")

    model, tokenizer = load_peft_model_and_tokenizer(model_name, quantize)
    model.train()

    logger.info("Loading and pre-processing dataset")
    train_dataset, test_dataset = get_train_test_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

    mlflow.end_run()
