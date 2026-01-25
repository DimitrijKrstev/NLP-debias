import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel


def evaluate_model(
    model: PreTrainedModel, dataloader: DataLoader
) -> tuple[float, float]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted"
    )

    model.train()
    return accuracy, f1
