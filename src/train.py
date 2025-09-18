import logging

import mlflow
import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from constants import OUTPUT_DIR
from src.dataset.preprocess import create_debiasing_prompt, get_train_test_dataset
from src.utils import load_peft_model_and_tokenizer

logger = logging.getLogger(__name__)


def train_model(
    quantize: bool,
    mlflow_experiment: str,
    model_name: str,
):
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=f"{model_name}-debiasing")

    model, tokenizer = load_peft_model_and_tokenizer(model_name, quantize)

    logger.info("Loading and pre-processing dataset")
    train_dataset, test_dataset = get_train_test_dataset(tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
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
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    mlflow.end_run()


def debias_text(text: str, model, tokenizer, max_length: int = 512):
    prompt = create_debiasing_prompt(text)

    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=max_length, truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return generated_text.strip()


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        max_steps=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=600,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="qwen3-debiasing",
    )
