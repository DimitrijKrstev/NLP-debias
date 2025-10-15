from logging import INFO, basicConfig, getLogger

import typer

from dataset.download import download_wnc
from evaluation.eval_model import evaluate_model
from train import train_model

app = typer.Typer()
from judge.llm_judge import run_rlhf_training

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
    model_tokenizer_path: str = "output/",
    model_name: str = "Qwen/Qwen3-4B",
) -> None:
    logger.info(f"Evaluating model: {model_name}")
    evaluate_model(model_tokenizer_path, model_name)


def train_rl_model_command(args):
    run_rlhf_training(args.model_tokenizer_path, args.llm_model)


def train_rl_model_command(args):
    run_rlhf_training(args.model_tokenizer_path, args.llm_model)


if __name__ == "__main__":
    app()
