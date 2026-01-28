from logging import getLogger
from pathlib import Path

import mlflow

from bias_classifier.constants import (
    BATCH_SIZE,
    EVAL_STEPS,
    LEARNING_RATE,
    MAX_TRAIN_SAMPLES,
    MODEL_NAME,
    MODEL_OUTPUT_DIR,
    NUM_EPOCHS,
    OUTPUT_DIR,
)
from bias_classifier.dataset import get_classification_dataset
from bias_classifier.eval import evaluate_classifier
from bias_classifier.train import train_classifier
from bias_classifier.utils import load_model, load_tokenizer, load_trained_model
from dataset.enums import DatasetSplit

logger = getLogger(__name__)


def run_train_bias_classifier(
    model_name: str = MODEL_NAME,
    mlflow_experiment: str = "Bias-Classifier",
) -> None:
    mlflow.set_experiment(mlflow_experiment)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{model_name}-bias-classifier"):
        mlflow.log_params({
            "model_name": model_name,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_train_samples": MAX_TRAIN_SAMPLES or "all",
            "eval_steps": EVAL_STEPS or "per_epoch",
        })
        mlflow.set_tag("task", "bias_classification")

        tokenizer = load_tokenizer(model_name)
        model = load_model(model_name)

        logger.info("Loading and transforming datasets...")
        train_dataset = get_classification_dataset(
            DatasetSplit.TRAIN, tokenizer, max_samples=MAX_TRAIN_SAMPLES
        )
        val_dataset = get_classification_dataset(DatasetSplit.VALIDATION, tokenizer)
        test_dataset = get_classification_dataset(DatasetSplit.TEST, tokenizer)

        trainer = train_classifier(model, tokenizer, train_dataset, val_dataset)

        logger.info("Running final evaluation on test set...")
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        mlflow.log_metrics({
            "test_accuracy": test_metrics["test_accuracy"],
            "test_f1": test_metrics["test_f1"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
        })

        mlflow.log_artifact(str(MODEL_OUTPUT_DIR))

    logger.info("Training complete!")


def run_eval_bias_classifier(
    model_path: str = str(MODEL_OUTPUT_DIR),
    mlflow_experiment: str = "Bias-Classifier",
) -> None:
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"eval-{Path(model_path).name}"):
        mlflow.log_param("model_path", model_path)
        mlflow.set_tag("task", "evaluation")

        model, tokenizer = load_trained_model(model_path)

        test_dataset = get_classification_dataset(DatasetSplit.TEST, tokenizer)

        results = evaluate_classifier(model, tokenizer, test_dataset)

        mlflow.log_metrics({
            "test_accuracy": results["eval_accuracy"],
            "test_f1": results["eval_f1"],
            "test_precision": results["eval_precision"],
            "test_recall": results["eval_recall"],
        })

    logger.info("Evaluation complete!")
