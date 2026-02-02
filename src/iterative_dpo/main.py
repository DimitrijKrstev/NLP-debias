from logging import getLogger
from pathlib import Path

import mlflow
from datasets import Dataset  # type: ignore
from tqdm import tqdm
from trl.trainer.dpo_trainer import DPOTrainer

from constants import TRAIN_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType, WNCColumn
from dataset.preprocess import get_dataset_split
from iterative_dpo.constants import ITERATIVE_DPO_OUTPUT_DIR
from iterative_dpo.utils import (
    create_preference_pairs,
    generate_responses_with_temperatures,
    get_dpo_config,
)
from unsloth_load import load_unsloth_model

logger = getLogger(__name__)


def run_iterative_dpo_training(
    model_name: str,
    model_tokenizer_path: str | None,
    mlflow_experiment: str,
    judge_model_name: str,
    quantize: bool,
    sample_train_batch: int,
    output_dir: str | None = None,
    previous_checkpoint_path: str | None = None,
) -> None:
    mlflow.set_experiment(mlflow_experiment)

    output_location = output_dir or ITERATIVE_DPO_OUTPUT_DIR

    logger.info(f"Loading base model: {model_name}")
    model, tokenizer = load_unsloth_model(model_name, quantize)

    if model_tokenizer_path:
        logger.info(f"Loading adapters from checkpoint: {model_tokenizer_path}")
        model.load_adapter(model_tokenizer_path, adapter_name="checkpoint")
        model.set_adapter("checkpoint")

    dataset = get_dataset_split(DatasetSplit.TRAIN, TokenizationType.DPO, tokenizer)

    previous = 0
    total_iterations = (len(dataset) + sample_train_batch - 1) // sample_train_batch

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": model_name,
                "judge_model_name": judge_model_name,
                "sample_train_batch": sample_train_batch,
                "quantize": quantize,
            }
        )

        pbar = tqdm(total=total_iterations, desc="Iterative DPO Training")
        iteration = 0

        while previous < len(dataset):
            batch = dataset[previous : previous + sample_train_batch]

            all_responses = [
                generate_responses_with_temperatures(model, tokenizer, formatted_prompt)
                for formatted_prompt in tqdm(
                    batch["formatted_prompt"],
                    desc=f"Generating responses (iter {iteration})",
                    leave=False,
                )
            ]

            preference_pairs = create_preference_pairs(
                batch["prompt"],
                batch["formatted_prompt"],
                batch[WNCColumn.BIASED],
                batch[WNCColumn.NEUTRAL],
                all_responses,
                judge_model_name,
                iteration,
            )

            dataset_dict = {
                "prompt": [pair.formatted_prompt for pair in preference_pairs],
                "chosen": [pair.chosen for pair in preference_pairs],
                "rejected": [pair.rejected for pair in preference_pairs],
            }
            train_dataset = Dataset.from_dict(dataset_dict)

            dpo_trainer = DPOTrainer(
                model=model,
                processing_class=tokenizer,
                args=get_dpo_config(model_name, output_dir or TRAIN_OUTPUT_DIR),
                train_dataset=train_dataset,
            )

            train_result = dpo_trainer.train(
                resume_from_checkpoint=previous_checkpoint_path
            )
            previous_checkpoint_path = None

            train_loss = train_result.metrics.get("train_loss", 0.0)
            logger.info(f"Iteration {iteration} - train_loss: {train_loss:.4f}")

            checkpoint_dir = (
                Path(output_location) / "iterative_dpo" / f"checkpoint-iter-{iteration}"
            )
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            logger.info(f"Model checkpoint saved to {checkpoint_dir}")

            previous += sample_train_batch
            iteration += 1
            pbar.update(1)

        pbar.close()

    model.save_pretrained(output_location)
    tokenizer.save_pretrained(output_location)

    logger.info("Iterative DPO training completed")
