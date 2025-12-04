import csv
from logging import getLogger

from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.enums import DatasetSplit, TokenizationType, WNCColumn
from dataset.preprocess import get_dataset_split
from evaluation.constants import EVAL_RESULT_CSV
from evaluation.utils import compute_metrics, debias_text
from utils import load_model, load_tokenizer

logger = getLogger(__name__)


def evaluate_model(model_tokenizer_path: str, model_name: str) -> None:
    tokenizer = load_tokenizer(model_tokenizer_path)
    base_model = load_model(model_name, True)

    logger.info(f"Loading LoRA adapter from {model_tokenizer_path}")
    model = PeftModel.from_pretrained(base_model, model_tokenizer_path)
    model.eval()

    test_dataset = get_dataset_split(DatasetSplit.TEST, TokenizationType.SFT, tokenizer)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # type: ignore[arg-type]

    all_predictions = []
    with open(EVAL_RESULT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([WNCColumn.BIASED, WNCColumn.NEUTRAL, "predicted"])

        for batch in tqdm(loader, desc="Evaluating model"):
            biased_texts = batch[WNCColumn.BIASED]
            neutral_texts = batch[WNCColumn.NEUTRAL]

            predicted_texts = debias_text(biased_texts, model, tokenizer)

            for biased, neutral, predicted in zip(
                biased_texts, neutral_texts, predicted_texts
            ):
                writer.writerow([biased, neutral, predicted])

            all_predictions.extend(predicted_texts)

    results = compute_metrics(all_predictions, test_dataset[WNCColumn.NEUTRAL])

    logger.info(
        "Evaluation Metrics:\n\n"
        f"BLEU:           {results.bleu:.2f}\n"
        f"METEOR:         {results.meteor:.2f}\n"
        f"ROUGE-L:        {results.rougeL:.2f}\n"
        f"Semantic Sim:   {results.semantic_similarity:.2f}\n"
        f"BERTScore-F1:   {results.bertscore_f1:.2f}"
        f"Judge Score:    {results.judge_score:.2f}"
    )
