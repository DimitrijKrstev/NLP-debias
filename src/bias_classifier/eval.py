from logging import getLogger
from typing import Any

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction, PreTrainedModel, PreTrainedTokenizer, Trainer

from bias_classifier.constants import BATCH_SIZE, ID2LABEL

logger = getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    precision = precision_score(labels, predictions, average="binary")
    recall = recall_score(labels, predictions, average="binary")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate_classifier(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> dict[str, Any]:
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Evaluating model...")
    results = trainer.evaluate()

    logger.info(f"Evaluation results: {results}")
    return results


def get_predictions_with_labels(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
) -> list[dict[str, Any]]:
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)

    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids

    results = []
    for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
        results.append({
            "predicted": ID2LABEL[pred],
            "actual": ID2LABEL[true],
            "correct": pred == true,
        })

    return results
