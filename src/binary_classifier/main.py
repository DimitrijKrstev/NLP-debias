import mlflow
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from binary_classifier.train import train_binary_classifier
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from utils import load_peft_classifcation_model, load_tokenizer


def run_train_binary_classifier(
    model_name: str, model_tokenizer_path: str, mlflow_experiment: str, quantize: bool
) -> None:
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        tokenizer = load_tokenizer(model_tokenizer_path)
        model = load_peft_classifcation_model(model_name, quantize)

        full_dataset = get_dataset_split(
            DatasetSplit.FULL, TokenizationType.BINARY_CLASSIFICATION, tokenizer
        )

        batch_size = 8
        learning_rate = 5e-5
        num_epochs = 3

        mlflow.log_params(
            {
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "quantize": quantize,
            }
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataloader = DataLoader(
            full_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator  # type: ignore
        )
        val_dataloader = DataLoader(
            full_dataset[-600:-300], batch_size=batch_size, shuffle=False, collate_fn=data_collator  # type: ignore
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        loss_fn = CrossEntropyLoss()

        model.train()
        train_binary_classifier(
            model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs
        )
