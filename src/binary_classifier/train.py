from typing import Any

import mlflow
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from binary_classifier.eval import evaluate_model


def train_binary_classifier(
    model: PreTrainedModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Any,
    num_epochs: int,
):
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} / {num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            progress_bar.set_postfix({"loss": loss.item()})

            if global_step % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        val_accuracy, val_f1 = evaluate_model(model, val_dataloader)
        print(f"Validation accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        mlflow.log_metrics(
            {
                "epoch_train_loss": avg_epoch_loss,
                "val_accuracy": float(val_accuracy),
                "val_f1": float(val_f1),
            },
            step=epoch,
        )

    print("Training completed!")
