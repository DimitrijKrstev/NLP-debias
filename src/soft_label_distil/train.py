import logging
from os import environ
from pathlib import Path

import mlflow

from constants import DISTIL_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from soft_label_distil.trainer import DistillationTrainer
from soft_label_distil.utils import (
    get_distillation_training_args,
    load_teacher_responses,
)
from utils import load_unsloth_model

environ["TOKENIZERS_PARALLELISM"] = "false"
environ["UNSLOTH_RETURN_LOGITS"] = "1"

logger = logging.getLogger(__name__)


def train_distillation_model(
    quantize: bool,
    mlflow_experiment: str,
    model_name: str,
    teacher_responses_file: str,
    teacher_model_name: str = "qwen/qwen3-235b-a22b",
    temperature: float = 1.5,
    alpha: float = 0.5,
) -> None:
    mlflow.set_experiment(mlflow_experiment)
    load_teacher_responses(Path(teacher_responses_file))

    with mlflow.start_run(run_name=f"{model_name}-distillation"):
        training_args = get_distillation_training_args()

        mlflow.log_params(
            {
                "model_name": model_name,
                "teacher_model_name": teacher_model_name,
                "quantize": quantize,
                "temperature": temperature,
                "alpha": alpha,
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "warmup_ratio": training_args.warmup_ratio,
                "weight_decay": training_args.weight_decay,
                "output_dir": str(DISTIL_OUTPUT_DIR),
            }
        )
        mlflow.set_tag("task", "training")
        mlflow.set_tag("training_type", "distillation")

        model, tokenizer = load_unsloth_model(model_name, quantize)
        model.train()

        train_dataset = get_dataset_split(
            DatasetSplit.TRAIN, TokenizationType.DISTIL, tokenizer
        )
        val_dataset = get_dataset_split(
            DatasetSplit.VALIDATION, TokenizationType.DISTIL, tokenizer
        )

        mlflow.log_params(
            {
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
            }
        )

        trainer = DistillationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            teacher_model_name=teacher_model_name,
            temperature=temperature,
            alpha=alpha,
            data_collator=None,
        )

        logger.info(
            f"Starting distillation training with teacher={teacher_model_name}, "
            f"temperature={temperature}, alpha={alpha}..."
        )

        trainer.train()

        logger.info(f"Saving model and tokenizer to {DISTIL_OUTPUT_DIR}")
        trainer.save_model(DISTIL_OUTPUT_DIR.as_posix())
        tokenizer.save_pretrained(DISTIL_OUTPUT_DIR)

        mlflow.log_param("final_checkpoint", str(DISTIL_OUTPUT_DIR))

    logger.info("Distillation training complete!")
