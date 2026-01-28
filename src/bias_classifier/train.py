from logging import getLogger

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from bias_classifier.constants import (
    BATCH_SIZE,
    EVAL_STEPS,
    LEARNING_RATE,
    MODEL_OUTPUT_DIR,
    NUM_EPOCHS,
    WARMUP_RATIO,
)
from bias_classifier.eval import compute_metrics

logger = getLogger(__name__)


def get_training_args(output_dir: str = str(MODEL_OUTPUT_DIR)) -> TrainingArguments:
    eval_strategy = "steps" if EVAL_STEPS else "epoch"
    save_strategy = "steps" if EVAL_STEPS else "epoch"

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy=eval_strategy,
        eval_steps=EVAL_STEPS,
        save_strategy=save_strategy,
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="mlflow",
        fp16=True,
        max_grad_norm=1.0,
    )


def train_classifier(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> Trainer:
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {MODEL_OUTPUT_DIR}")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    return trainer
