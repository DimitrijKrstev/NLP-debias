from logging import INFO, basicConfig, getLogger
from pathlib import Path

import typer
from transformers import AutoTokenizer, PreTrainedTokenizer

basicConfig(level=INFO, format="%(levelname)s - %(filename)s:%(lineno)d - %(message)s")

from bias_classifier.constants import MODEL_NAME as BIAS_CLASSIFIER_MODEL_NAME
from bias_classifier.constants import MODEL_OUTPUT_DIR as BIAS_CLASSIFIER_OUTPUT_DIR
from bias_classifier.main import run_eval_bias_classifier, run_train_bias_classifier
from binary_classifier.main import run_train_binary_classifier
from constants import DISTIL_OUTPUT_DIR
from dataset.download import download_wnc
from dataset.enums import TokenizationType
from evaluation.eval_model import evaluate_model
from iterative_dpo.main import run_iterative_dpo_training
from judge.main import get_judge_score
from rl.main import run_grpo_training
from sft.train import train_model
from soft_label_distil.train import train_distillation_model
from soft_label_distil.utils import (
    get_teacher_logprobs,
    load_teacher_responses,
)

app = typer.Typer(pretty_exceptions_enable=False)

logger = getLogger(__name__)


@app.command()
def download_dataset() -> None:
    logger.info("Downloading dataset...")
    download_wnc()
    logger.info("Dataset downloaded successfully.")


@app.command()
def sft_train_model(
    quantize: bool = True,
    mlflow_experiment: str = "SFT-NLP-Debias",
    model_name: str = "Qwen/Qwen3-1.7B",
) -> None:
    logger.info(f"Training model: {model_name} (quantize={quantize})")
    train_model(quantize, mlflow_experiment, model_name)


@app.command()
def eval_model(
    model_tokenizer_path: str = "Qwen/Qwen3-1.7B",
    model_name: str = "Qwen/Qwen3-1.7B",
    judge_model_name: str = "google/gemini-3-flash-preview",
    tokenization_type: TokenizationType = TokenizationType.EVAL,
    mlflow_experiment: str = "NLP-Debias-Eval",
    existing_evalution_csv: str | None = None,
) -> None:
    logger.info(f"Evaluating model: {model_name}")
    evaluate_model(
        model_tokenizer_path,
        model_name,
        judge_model_name,
        tokenization_type,
        mlflow_experiment,
        Path(existing_evalution_csv) if existing_evalution_csv is not None else None,
    )


@app.command()
def train_rl_model(
    model_name: str = "Qwen/Qwen3-1.7B",
    mflow_experiment: str = "RL-NLP-Debias",
    quantize: bool = True,
) -> None:
    run_grpo_training(
        model_name,
        mflow_experiment,
        quantize,
    )


@app.command()
def test_judge(open_ai_remote_model_name: str = "gpt-5-mini") -> None:
    logger.info(
        get_judge_score(
            "during the campaign , controversy erupted over alleged differences between pali ##n ' s positions as a gubernatorial candidate and her position as a vice - presidential candidate .",
            "<|im_end|>",
            "during the campaign, some pointed out alleged differences between pali's positions as a gubernatorial candidate and her position as a vice - presidential candidate.",
            open_ai_remote_model_name,
            None,
        )
    )


@app.command()
def test_logprobs(
    remote_model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
) -> None:
    logger.info(
        get_teacher_logprobs(
            "during the campaign , controversy erupted over alleged differences between pali ##n ' s positions",
            remote_model_name,
        )
    )


@app.command()
def test_load_logprobs(
    logprobs_file: str = f"{DISTIL_OUTPUT_DIR}/teacher_responses.jsonl",
) -> None:
    logprobs = load_teacher_responses(Path(logprobs_file))
    logger.info(f"Loaded {len(logprobs)} logprobs from {logprobs_file}")

    if logprobs:
        first_biased_text = next(iter(logprobs.keys()))
        first_logprobs = logprobs[first_biased_text]
        logger.info(f"First entry - biased text: {first_biased_text}")
        logger.info(f"Number of tokens: {len(first_logprobs)}")
        logger.info(f"First token logprobs: {first_logprobs[0]}")


@app.command()
def distil_train_model(
    quantize: bool = True,
    mlflow_experiment: str = "NLP-Debias-Distillation",
    model_name: str = "Qwen/Qwen3-1.7B",
    teacher_responses_file: str = f"{DISTIL_OUTPUT_DIR}/teacher_responses.jsonl",
    teacher_model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    temperature: float = 1.5,
    alpha: float = 0.5,
) -> None:
    logger.info(
        f"Training model via distillation: student={model_name}, teacher={teacher_model_name}, "
        f"temperature={temperature}, alpha={alpha}"
    )
    train_distillation_model(
        quantize=quantize,
        mlflow_experiment=mlflow_experiment,
        model_name=model_name,
        teacher_responses_file=teacher_responses_file,
        teacher_model_name=teacher_model_name,
        temperature=temperature,
        alpha=alpha,
    )


@app.command()
def train_iterative_dpo(
    model_name: str = "Qwen/Qwen3-1.7B",
    model_tokenizer_path: str | None = None,
    mlflow_experiment: str = "Iterative-DPO",
    judge_model_name: str = "gpt-5-mini",
    quantize: bool = True,
    batch_size: int = 25,
) -> None:
    run_iterative_dpo_training(
        model_name=model_name,
        model_tokenizer_path=model_tokenizer_path,
        mlflow_experiment=mlflow_experiment,
        judge_model_name=judge_model_name,
        quantize=quantize,
        sample_train_batch=batch_size,
    )


@app.command()
def test_tokenizer_match(
    model_1: str = "Qwen/Qwen3-1.7B",
    model_2: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
) -> None:
    tokenizer_1: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_1)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2)

    logger.info(tokenizer_1.vocab_size)
    logger.info(tokenizer_2.vocab_size)
    logger.info(tokenizer_1)
    logger.info(tokenizer_2)


@app.command()
def train_binary_classifier(
    model_name: str = "Qwen/Qwen3-1.7B",
    model_tokenizer_path: str = "Qwen/Qwen3-1.7B",
    mlflow_experiment: str = "Binary-Classifier",
    quantize: bool = True,
) -> None:
    run_train_binary_classifier(
        model_name, model_tokenizer_path, mlflow_experiment, quantize
    )


@app.command()
def train_bias_classifier(
    model_name: str = BIAS_CLASSIFIER_MODEL_NAME,
    mlflow_experiment: str = "Bias-Classifier",
) -> None:
    logger.info(f"Training bias classifier with {model_name}")
    run_train_bias_classifier(model_name, mlflow_experiment)


@app.command()
def eval_bias_classifier(
    model_path: str = str(BIAS_CLASSIFIER_OUTPUT_DIR),
    mlflow_experiment: str = "Bias-Classifier",
) -> None:
    logger.info(f"Evaluating bias classifier from {model_path}")
    run_eval_bias_classifier(model_path, mlflow_experiment)


if __name__ == "__main__":
    app()
