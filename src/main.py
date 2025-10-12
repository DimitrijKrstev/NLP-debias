import argparse
from logging import INFO, basicConfig, getLogger
from pathlib import Path

from dataset.download import download_wnc
from eval_model import evaluate_model
from train import train_model

basicConfig(level=INFO)
logger = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NLP Debias CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_ds_parser = subparsers.add_parser(
        "download-dataset", help="Download dataset"
    )
    download_ds_parser.set_defaults(func=download_dataset_command)

    train_model_parser = subparsers.add_parser("train-model", help="Train model")
    train_model_parser.set_defaults(func=train_model_command)
    train_model_parser.add_argument(
        "--quantize", action="store_true", help="Use 4-bit quantization"
    )
    train_model_parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="NLP-Debias",
        help="MLflow experiment name",
    )
    train_model_parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model name",
    )

    eval_model_parser = subparsers.add_parser("eval-model", help="Evaluate model")
    eval_model_parser.set_defaults(func=eval_model_command)
    eval_model_parser.add_argument(
        "--model-tokenizer-path",
        type=str,
        default="./models/output/",
        help="Path to the trained model and tokenizer",
    )
    eval_model_parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Path to the model name",
    )

    args = parser.parse_args()
    args.func(args)


def download_dataset_command(_):
    download_wnc()


def train_model_command(args):
    train_model(args.quantize, args.mlflow_experiment, args.model_name)


def eval_model_command(args):
    # evaluate_model(Path(__file__).parent.parent / args.model_tokenizer_path)
    evaluate_model(Path(args.model_tokenizer_path), args.model_name)


if __name__ == "__main__":
    main()
