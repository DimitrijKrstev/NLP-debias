import csv
import logging
from pathlib import Path
from typing import List, Optional

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
from dataset.preprocess import get_train_dataset, tokenize_data
from rl.main import run_rlhf_training
from rl.utils import remove_thinking_tags
from train import train_model
from utils import load_peft_model, load_tokenizer

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_grpo_outputs(
    grpo_model_path: str | Path,
    tokenizer_path: Optional[str | Path] = None,
    generation_batch_size: int = 8,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    output_csv: str | Path = "grpo_generated_outputs.csv",
    limit: Optional[int] = None,
) -> Path:

    grpo_model_path = Path(grpo_model_path)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path is None:
        tokenizer = load_tokenizer(grpo_model_path)
    else:
        tokenizer = load_tokenizer(tokenizer_path)

    model = load_peft_model(str(grpo_model_path), quantize=False)
    model.to(device)
    model.eval()

    rl_dataset = get_train_dataset(tokenizer, is_rl=True)
    logger.info(
        f"Generating GRPO outputs for {len(rl_dataset)} samples (limit={limit})"
    )

    rows: list[dict[str, str]] = []
    count = 0

    batch: List[dict] = []
    for sample in rl_dataset:
        prompt = sample["prompt"]
        biased = sample.get(WNCColumn.BIASED, "")
        reference = sample.get(WNCColumn.NEUTRAL, "")

        batch.append({"prompt": prompt, "biased": biased, "reference": reference})

        if len(batch) >= generation_batch_size:
            process_batch(
                batch,
                model,
                tokenizer,
                device,
                rows,
                max_input_length,
                max_new_tokens,
                temperature,
                top_p,
            )
            batch = []

        count += 1
        if limit is not None and count >= limit:
            break

    if batch:
        process_batch(
            batch,
            model,
            tokenizer,
            device,
            rows,
            max_input_length,
            max_new_tokens,
            temperature,
            top_p,
        )

    with open(out_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["biased_text", "grpo_output", "reference_text"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "biased_text": r["biased_text"],
                    "grpo_output": r["grpo_output"],
                    "reference_text": r["reference_text"],
                }
            )

    logger.info(f"Wrote {len(rows)} generated rows to {out_path}")
    return out_path


def process_batch(
    batch,
    model,
    tokenizer,
    device,
    rows,
    max_input_length,
    max_new_tokens,
    temperature,
    top_p,
):

    prompts = [b["prompt"] for b in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_input_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_lengths = inputs["input_ids"].shape[1]

    for i, out_ids in enumerate(generated):
        completion_ids = out_ids[input_lengths:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        completion = remove_thinking_tags(completion)
        rows.append(
            {
                "biased_text": batch[i]["biased"],
                "grpo_output": completion,
                "reference_text": batch[i]["reference"],
            }
        )


def train_sft_on_generated(
    base_sft_checkpoint: str | Path,
    generated_csv: str | Path,
    output_dir: str | Path,
    per_device_train_batch_size: int = 8,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
) -> None:

    base_sft_checkpoint = str(base_sft_checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(base_sft_checkpoint)
    model = load_peft_model(base_sft_checkpoint, quantize=False)
    model.train()

    df = pd.read_csv(generated_csv)
    df = df.rename(
        columns={"biased_text": WNCColumn.BIASED, "grpo_output": WNCColumn.NEUTRAL}
    )
    df = df.dropna(subset=[WNCColumn.BIASED, WNCColumn.NEUTRAL]).reset_index(drop=True)

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
        fp16=torch.cuda.is_available(),
        bf16=not torch.cuda.is_available(),
        run_name="sft-after-grpo",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    logger.info("Starting SFT fine-tuning on GRPO outputs...")
    trainer.train()
    logger.info(f"Saving stabilized SFT model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Stabilization SFT complete.")


def run_hybrid_sft_grpo_sft(
    mlflow_experiment: str,
    quantize: bool = False,
    generated_output_csv: str | Path = "grpo_generated_outputs.csv",
    sft_after_grpo_dir: str | Path = Path(TRAIN_OUTPUT_DIR) / "sft-after-grpo",
    generation_limit: Optional[int] = None,
):
    logger.info("STEP 1: Running initial SFT training (supervised).")

    train_model(
        quantize=quantize, mlflow_experiment=mlflow_experiment, model_name="sft-initial"
    )
    initial_sft_checkpoint = TRAIN_OUTPUT_DIR

    logger.info("STEP 2: Running GRPO starting from SFT checkpoint.")

    run_rlhf_training(str(initial_sft_checkpoint), mlflow_experiment, quantize)
    grpo_model_path = RL_OUTPUT_DIR / "grpo-debiasing-model"

    logger.info("STEP 3: Generating GRPO outputs for supervised stabilization dataset.")
    gen_csv = generate_grpo_outputs(
        grpo_model_path=grpo_model_path,
        tokenizer_path=initial_sft_checkpoint,
        generation_batch_size=8,
        max_input_length=256,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        output_csv=generated_output_csv,
        limit=generation_limit,
    )

    logger.info("STEP 4: Fine-tuning SFT on GRPO outputs (stabilization).")
    train_sft_on_generated(
        base_sft_checkpoint=grpo_model_path,
        generated_csv=gen_csv,
        output_dir=sft_after_grpo_dir,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=1e-4,
    )

    logger.info("Hybrid SFT-GRPO-SFT pipeline complete.")
