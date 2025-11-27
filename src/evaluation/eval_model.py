import csv
from logging import getLogger

from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.enums import DatasetSplit, TokenizationType, WNCColumn
from dataset.preprocess import get_dataset_split
from evaluation.utils import compute_metrics, debias_text
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
    test_dataset = get_dataset_split(DatasetSplit.TEST, TokenizationType.SFT, tokenizer)

    logger.info(f"Evaluating on {len(test_dataset)} examples...")

    batch_size = 16
    loader = DataLoader(test_dataset, batch_size=batch_size)  # type: ignore[arg-type]

    csv_file = "wnc_predictions.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["biased", "neutral_ref", "predicted"])

        all_predictions = []
        all_references = []

        for batch in tqdm(loader):
            biased_texts = batch[WNCColumn.BIASED]
            neutral_texts = batch[WNCColumn.NEUTRAL]

            prompts = [text for text in biased_texts]
            predicted_texts = debias_text(prompts, model, tokenizer)

            for b, r, p in zip(biased_texts, neutral_texts, predicted_texts):
                writer.writerow([b, r, p])

            all_predictions.extend(predicted_texts)
            all_references.extend(neutral_texts)

    results = compute_metrics(all_predictions, all_references)

    logger.info(
        "Evaluation Metrics:\n"
        f"  BLEU:           {results.bleu:.2f}\n"
        f"  METEOR:         {results.meteor:.2f}\n"
        f"  ROUGE-L:        {results.rougeL:.2f}\n"
        f"  Semantic Sim:   {results.semantic_similarity:.2f}\n"
        f"  BERTScore-F1:   {results.bertscore_f1:.2f}"
    )
