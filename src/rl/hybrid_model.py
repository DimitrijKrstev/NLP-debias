import gc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset  # type: ignore[import-untyped]
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from constants import RL_OUTPUT_DIR, TRAIN_OUTPUT_DIR
from dataset.constants import WNCColumn
from dataset.preprocess import tokenize_data
from rl.main import run_rlhf_training
from train import train_model
from utils import load_peft_model, load_tokenizer

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_sft_on_generated(
    base_sft_checkpoint: str | Path,
    generated_csv: str | Path,
    output_dir: str | Path,
    per_device_train_batch_size: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    min_score: float = 0.7,
) -> None:
    base_sft_checkpoint = str(base_sft_checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(base_sft_checkpoint)
    model = load_peft_model(base_sft_checkpoint, quantize=True)
    model.train()

    df = pd.read_csv(generated_csv)

    df = df.rename(
        columns={"biased_text": WNCColumn.BIASED, "model_output": WNCColumn.NEUTRAL}
    )

    df = df.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL, "score"]).reset_index(
        drop=True
    )

    df = df[df["score"] >= min_score].reset_index(drop=True)
    logger.info(f"Using {len(df)} examples with score >= {min_score} for SFT.")

    if len(df) == 0:
        logger.warning("No examples meet the score threshold. Aborting SFT training.")
        return

    dataset = Dataset.from_pandas(df)

    tokenized = dataset.map(
        lambda batch: tokenize_data(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # ✅ Use 8-bit optimizer for memory efficiency
        run_name="sft-after-grpo",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting SFT fine-tuning on high-score GRPO outputs...")
    trainer.train()
    logger.info(f"Saving stabilized SFT model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Stabilization SFT complete.")


def run_hybrid_sft_grpo_sft(
    mlflow_experiment: str,
    quantize: bool = True,
    sft_after_grpo_dir: str | Path = Path(TRAIN_OUTPUT_DIR) / "sft-after-grpo",
    generation_limit: Optional[int] = None,
):
    logger.info("STEP 1: Running initial SFT training (supervised).")

    train_model(
        quantize=quantize,
        mlflow_experiment=mlflow_experiment,
        model_name="Qwen/Qwen3-4B",
    )
    initial_sft_checkpoint = max(
        TRAIN_OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )

    logger.info("Clearing GPU memory before GRPO...")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("STEP 2: Running GRPO starting from SFT checkpoint.")

    run_rlhf_training(str(initial_sft_checkpoint), mlflow_experiment, quantize)
    grpo_model_path = RL_OUTPUT_DIR / "grpo-debiasing-model"
    generated_output_csv = RL_OUTPUT_DIR / "judge_scores.csv"

    logger.info("Clearing GPU memory before stabilization SFT...")
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("STEP 3: Fine-tuning SFT on GRPO outputs (stabilization).")
    train_sft_on_generated(
        base_sft_checkpoint=grpo_model_path,
        generated_csv=generated_output_csv,
        output_dir=sft_after_grpo_dir,
    )

    logger.info("Hybrid SFT-GRPO-SFT pipeline complete.")
