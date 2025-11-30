from logging import INFO, basicConfig, getLogger
from typing import Optional

import typer

from constants import TRAIN_OUTPUT_DIR
from dataset.download import download_wnc
from evaluation.eval_model import evaluate_model
from rl.hybrid_model import run_hybrid_sft_grpo_sft
from rl.main import run_rlhf_training
from rl.utils import get_judge_score
from train import train_model

app = typer.Typer(pretty_exceptions_enable=False)

basicConfig(level=INFO)
logger = getLogger(__name__)


@app.command()
def download_dataset() -> None:
    logger.info("Downloading dataset...")
    download_wnc()
    logger.info("Dataset downloaded successfully.")


@app.command()
def sft_train_model(
    quantize: bool = True,
    mlflow_experiment: str = "NLP-Debias",
    model_name: str = "Qwen/Qwen3-4B",
) -> None:
    logger.info(f"Training model: {model_name} (quantize={quantize})")
    train_model(quantize, mlflow_experiment, model_name)


@app.command()
def eval_model(
    model_tokenizer_path: str = "grpo-debiasing-model/checkpoint-2200",
    model_name: str = "Qwen/Qwen3-4B",
) -> None:
    logger.info(f"Evaluating model: {model_name}")
    evaluate_model(model_tokenizer_path, model_name)


@app.command()
def train_rl_model(
    model_name: str = "Qwen/Qwen3-4B",
    mflow_experiment: str = "RL-NLP-Debias",
    quantize: bool = True,
) -> None:
    run_rlhf_training(
        model_name,
        mflow_experiment,
        quantize,
    )


@app.command()
def train_hybrid_model(
    mlflow_experiment: str = "Hybrid-NLP-Debias",
    quantize: bool = False,
    sft_after_grpo_dir: str = "sft-after-grpo",
    generation_limit: Optional[int] = 2,
) -> None:
    run_hybrid_sft_grpo_sft(
        mlflow_experiment,
        quantize,
        TRAIN_OUTPUT_DIR / sft_after_grpo_dir,
        generation_limit,
    )


@app.command()
def test_openai(open_ai_remote_model_name: str = "gpt-5-mini") -> None:
    logger.info(
        get_judge_score(
            "during the campaign , controversy erupted over alleged differences between pali ##n ' s positions as a gubernatorial candidate and her position as a vice - presidential candidate .",
            "during the campaign , some pointed out alleged differences between pali ##n ' s positions as a gubernatorial candidate and her position as a vice - presidential candidate .	",
            "during the campaign, some pointed out alleged differences between pali's positions as a gubernatorial candidate and her position as a vice - presidential candidate.",
            open_ai_remote_model_name,
        )
    )


if __name__ == "__main__":
    app()
