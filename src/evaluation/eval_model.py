import csv
from logging import getLogger
from pathlib import Path
from typing import Any

import mlflow
from datasets import Dataset  # type: ignore[import-untyped]
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from constants import JUDGE_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType, WNCColumn
from dataset.preprocess import get_dataset_split
from evaluation.constants import BATCH_SIZE, EVAL_DIR
from evaluation.utils import compute_metrics, debias_text
from utils import load_model, load_tokenizer

logger = getLogger(__name__)


def evaluate_model(
    model_tokenizer_path: str,
    model_name: str,
    judge_model_name: str,
    tokenization_type: TokenizationType,
    mlflow_experiment: str,
    existing_evalution_csv: Path | None,
) -> None:
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"eval_{model_tokenizer_path.replace('/', '_')}"):
        mlflow.log_params(
            {
                "checkpoint_path": model_tokenizer_path,
                "base_model_name": model_name,
                "judge_model_name": judge_model_name,
            }
        )
        mlflow.set_tag("task", "evaluation")

        tokenizer = load_tokenizer(model_name)

        if existing_evalution_csv is not None:
            predictions_csv_path = Path(existing_evalution_csv)
            all_predictions = []
            with open(
                predictions_csv_path, mode="r", newline="", encoding="utf-8"
            ) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    all_predictions.append(row[2])

            test_dataset = get_dataset_split(
                DatasetSplit.TEST, tokenization_type, tokenizer
            )

        else:
            try:
                base_model = load_model(model_name, True)
                model = PeftModel.from_pretrained(base_model, model_tokenizer_path)
                logger.info("Successfully loaded as PEFT model")
            except Exception as e:
                logger.error(f"Failed to load as PEFT model: {e}")
                config = AutoConfig.from_pretrained(
                    model_tokenizer_path, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
                    model_tokenizer_path,
                    config=config,
                    load_in_4bit=True,
                    device_map="auto",
                )

            model.eval()

            test_dataset = get_dataset_split(
                DatasetSplit.TEST, tokenization_type, tokenizer
            )
            mlflow.log_param("test_dataset_size", len(test_dataset))

            all_predictions, predictions_csv_path = _generate_predictions(
                model,
                tokenizer,
                test_dataset,
                model_tokenizer_path,
            )

        results = compute_metrics(
            test_dataset[WNCColumn.BIASED],
            test_dataset[WNCColumn.NEUTRAL],
            all_predictions,
            judge_model_name,
            model_tokenizer_path,
        )

        mlflow.log_metrics(results.to_dict())
        mlflow.log_artifact(str(predictions_csv_path))

        judge_csv_path = (
            JUDGE_OUTPUT_DIR / f"{model_tokenizer_path.replace('/', '_')}.csv"
        )
        if judge_csv_path.exists():
            mlflow.log_artifact(str(judge_csv_path))

        logger.info(f"{results.to_dict()}")


def _generate_predictions(
    model: Any,
    tokenizer: Any,
    test_dataset: Dataset,
    model_tokenizer_path: str,
) -> tuple[list[str], Path]:
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # type: ignore
    all_predictions = []

    eval_results_path = (
        EVAL_DIR
        / f"{model_tokenizer_path.replace(".", "").replace("/", "_").replace(" ", "_")}.csv"
    )
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_results_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([WNCColumn.BIASED, WNCColumn.NEUTRAL, "predicted"])

        for batch in tqdm(loader, desc="Evaluating model"):
            biased_texts = batch[WNCColumn.BIASED]
            neutral_texts = batch[WNCColumn.NEUTRAL]

            predicted_texts = debias_text(biased_texts, model, tokenizer)

            for biased, neutral, predicted in zip(
                biased_texts, neutral_texts, predicted_texts
            ):
                writer.writerow([biased, neutral, predicted])

            all_predictions.extend(predicted_texts)

    return all_predictions, eval_results_path
