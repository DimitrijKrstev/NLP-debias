import logging
from os import environ

import mlflow
from transformers import DataCollatorForLanguageModeling, Trainer

from constants import TRAIN_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from sft.utils import get_training_args
from utils import load_peft_model, load_tokenizer

environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def train_model(quantize: bool, mlflow_experiment: str, model_name: str) -> None:
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"{model_name}-debiasing"):
        training_args = get_training_args()

        mlflow.log_params(
            {
                "model_name": model_name,
                "quantize": quantize,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "warmup_ratio": training_args.warmup_ratio,
                "weight_decay": training_args.weight_decay,
                "output_dir": str(TRAIN_OUTPUT_DIR),
            }
        )
        mlflow.set_tag("task", "training")
        mlflow.set_tag("training_type", "sft")

        model = load_peft_model(model_name, quantize)
        tokenizer = load_tokenizer(model_name)
        model.train()

        train_dataset = get_dataset_split(
            DatasetSplit.TRAIN, TokenizationType.SFT, tokenizer
        )
        val_dataset = get_dataset_split(
            DatasetSplit.VALIDATION, TokenizationType.SFT, tokenizer
        )

        mlflow.log_params(
            {
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
            }
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
        tokenizer.save_pretrained(TRAIN_OUTPUT_DIR)

        mlflow.log_param("final_checkpoint", str(TRAIN_OUTPUT_DIR))

    logger.info("Training complete!")
