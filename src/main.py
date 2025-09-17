import argparse
from logging import INFO, basicConfig, getLogger

from dataset.download import download_wnc

basicConfig(level=INFO)
logger = getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NLP Debias CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_ds_parser = subparsers.add_parser(
        "download-dataset", help="Download dataset"
    )
    download_ds_parser.set_defaults(func=download_dataset)

    args = parser.parse_args()
    args.func(args)


def download_dataset(_):
    download_wnc()


if __name__ == "__main__":
    main()
