from logging import getLogger
from pathlib import Path

import pandas as pd
from evaluate import load as evaluate_load
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import SFT_RL_CUTOFF
from dataset.constants import WNCColumn
from dataset.preprocess import get_test_dataset, load_wnc_from_csv
from utils import debias_text, load_model, load_tokenizer

logger = getLogger(__name__)


def evaluate_model(model_tokenizer_path: Path, model_name: str) -> dict:
    logger.info(f"Loading tokenizer from {model_tokenizer_path}")
    tokenizer = load_tokenizer(model_tokenizer_path)

    logger.info("Loading base model...")
    base_model = load_model(model_name, True)

    logger.info(f"Loading LoRA adapter from {model_tokenizer_path}")
    model = PeftModel.from_pretrained(base_model, model_tokenizer_path)

    model.eval()

    logger.info("Loading test dataset...")
    dataset = load_wnc_from_csv()
    dataset = dataset[: int(SFT_RL_CUTOFF * len(dataset))]
    test_dataset = get_test_dataset(tokenizer)

    predictions = []
    references = []
    biased_list = []

    logger.info(f"Evaluating on {len(test_dataset)} examples...")

    batch_size = 16
    loader = DataLoader(test_dataset, batch_size=batch_size)

    for batch in tqdm(loader):
        biased_texts = batch[WNCColumn.BIASED]
        neutral_texts = batch[WNCColumn.NEUTRAL]

        predicted_texts = debias_text(biased_texts, model, tokenizer)

        predictions.extend(predicted_texts)
        references.extend(neutral_texts)
        biased_list.extend(biased_texts)

    df = pd.DataFrame(
        {"biased": biased_list, "neutral_ref": references, "predicted": predictions}
    )

    df.to_csv("wnc_predictions.csv", index=False)

    bleu = evaluate_load("bleu")
    meteor = evaluate_load("meteor")
    rouge = evaluate_load("rouge")

    bleu_score = bleu.compute(
        predictions=predictions, references=[[r] for r in references]
    )
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = similarity_model.encode(predictions, convert_to_tensor=True)
    ref_emb = similarity_model.encode(references, convert_to_tensor=True)
    semantic_sim = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()

    results = {
        "bleu": bleu_score["bleu"] * 100,
        "meteor": meteor_score["meteor"] * 100,
        "rouge1": rouge_scores["rouge1"] * 100,
        "rouge2": rouge_scores["rouge2"] * 100,
        "rougeL": rouge_scores["rougeL"] * 100,
        "semantic_similarity": semantic_sim * 100,
        "num_examples": len(test_dataset),
    }

    logger.info(
        f"BLEU: {results['bleu']:.2f}, METEOR: {results['meteor']:.2f}, "
        f"ROUGE-L: {results['rougeL']:.2f}, Semantic Sim: {results['semantic_similarity']:.2f}"
    )

    return results
