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

    logger.info("Checking first training example...")
    first_example = train_dataset[0]
    input_ids = first_example["input_ids"]
    labels = first_example["labels"]

    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    decoded_labels = []
    for idx, (inp, lab) in enumerate(zip(input_ids, labels)):
        if lab != -100:
            decoded_labels.append(tokenizer.decode([lab]))
        else:
            decoded_labels.append("[MASK]")

    logger.info(f"Full input: {decoded_input}")
    logger.info(f"Labels (first 50): {' '.join(decoded_labels[:50])}")
    logger.info(f"Number of trainable tokens: {sum(1 for l in labels if l != -100)}")

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
