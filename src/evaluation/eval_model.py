from logging import getLogger
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.constants import WNCColumn
from dataset.preprocess import get_test_dataset
from utils import load_model, load_tokenizer
from evaluation.utils import compute_metrics

logger = getLogger(__name__)


def evaluate_model(model_tokenizer_path: Path, model_name: str, quantize: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(model_tokenizer_path)
    base_model = load_model(model_name, quantize)

    model = PeftModel.from_pretrained(base_model, model_tokenizer_path).to(device)
    model.eval()

    logger.info("Loading and pre-processing dataset")
    test_dataset = get_test_dataset(tokenizer)
    loader = DataLoader(test_dataset, batch_size=16)

    predictions, references, biased = [], [], []

    for batch in tqdm(loader):
        biased_texts = batch[WNCColumn.BIASED]
        neutral_texts = batch[WNCColumn.NEUTRAL]

        input_ids = biased["input_ids"].to(device)
        attention_mask = biased["attention_mask"].to(device)

        with torch.no_grad(), torch.cuda.amp.autocast("cuda"):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded_predicted = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_biased = tokenizer.batch_decode(
            biased_texts["input_ids"], skip_special_tokens=True
        )
        decoded_neutral = tokenizer.batch_decode(
            neutral_texts["input_ids"], skip_special_tokens=True
        )

        predicted_texts = [
            d[len(b) :].strip() if d.startswith(b) else d.strip()
            for d, b in zip(decoded_predicted, decoded_biased)
        ]

        predictions.extend(predicted_texts)
        references.extend(decoded_neutral)
        biased.extend(decoded_biased)

    df = pd.DataFrame(
        {"biased": biased, "neutral_ref": references, "predicted": predictions}
    )
    df.to_csv("wnc_predictions.csv", index=False)

    results = compute_metrics(predictions, references)

    logger.info(
        f"BLEU: {results['bleu']:.2f}, METEOR: {results['meteor']:.2f}, "
        f"ROUGE-L: {results['rougeL']:.2f}, Semantic Sim: {results['semantic_similarity']:.2f}"
    )
