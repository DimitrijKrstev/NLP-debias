from logging import getLogger

import pandas as pd
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.constants import WNCColumn
from dataset.preprocess import get_test_dataset
from evaluation.utils import (clean_output, compute_metrics, debias_text,
                              make_chat_prompt)
from utils import load_model, load_tokenizer

logger = getLogger(__name__)


def evaluate_model(model_tokenizer_path: str, model_name: str) -> None:
    logger.info(f"Loading tokenizer from {model_tokenizer_path}")
    tokenizer = load_tokenizer(model_tokenizer_path)

    logger.info("Loading base model...")
    base_model = load_model(model_name, True)

    logger.info(f"Loading LoRA adapter from {model_tokenizer_path}")
    model = PeftModel.from_pretrained(base_model, model_tokenizer_path)

    model.eval()

    logger.info("Loading test dataset...")
    test_dataset = get_test_dataset(tokenizer)

    predictions = []
    references = []
    biased_list = []

    logger.info(f"Evaluating on {len(test_dataset)} examples...")

    batch_size = 16
    loader = DataLoader(test_dataset, batch_size=batch_size)  # type: ignore[arg-type]

    for batch in tqdm(loader):
        biased_texts = batch[WNCColumn.BIASED]
        neutral_texts = batch[WNCColumn.NEUTRAL]

        prompts = [make_chat_prompt(text) for text in biased_texts]
        predicted_texts = debias_text(prompts, model, tokenizer)

        predicted_texts = [clean_output(t) for t in predicted_texts]

        predictions.extend(predicted_texts)
        references.extend(neutral_texts)
        biased_list.extend(biased_texts)

    df = pd.DataFrame(
        {"biased": biased_list, "neutral_ref": references, "predicted": predictions}
    )

    df.to_csv("wnc_predictions.csv", index=False)

    results = compute_metrics(predictions, references)

    logger.info(
        "Evaluation Metrics:\n"
        f"  BLEU:           {results.bleu:.2f}\n"
        f"  METEOR:         {results.meteor:.2f}\n"
        f"  ROUGE-L:        {results.rougeL:.2f}\n"
        f"  Semantic Sim:   {results.semantic_similarity:.2f}\n"
        f"  Perplexity:     {results.perplexity:.2f}\n"
        f"  BERTScore-F1:   {results.bertscore_f1:.2f}"
    )
