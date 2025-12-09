from transformers import TrainingArguments

from constants import TRAIN_OUTPUT_DIR


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=TRAIN_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="qwen3-debiasing",
        dataloader_num_workers=2,
        bf16=True,
    )
