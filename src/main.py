from logging import INFO, basicConfig, getLogger

import typer

from dataset.download import download_wnc
from evaluation.eval_model import evaluate_model
from rl.main import run_rlhf_training
from judge.main import get_judge_score
from sft.train import train_model

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
