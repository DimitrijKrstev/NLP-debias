from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from bias_classifier.constants import ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_NAME

logger = getLogger(__name__)


def load_model(model_name: str = MODEL_NAME) -> PreTrainedModel:
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    return model


def load_tokenizer(model_name: str = MODEL_NAME) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def load_trained_model(model_path: Path | str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger.info(f"Loading trained model from: {model_path}")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


class BiasClassifier:
    def __init__(self, model_path: Path | str):
        self.model, self.tokenizer = load_trained_model(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[int]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().tolist()

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for prob in probs:
            results.append({
                "neutral": prob[0].item(),
                "biased": prob[1].item(),
            })

        return results

    def get_bias_score(self, texts: list[str]) -> list[float]:
        probs = self.predict_proba(texts)
        return [p["biased"] for p in probs]

    def get_neutrality_score(self, texts: list[str]) -> list[float]:
        probs = self.predict_proba(texts)
        return [p["neutral"] for p in probs]
