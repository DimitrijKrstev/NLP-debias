from logging import getLogger
from pathlib import Path
import torch
import nltk
import pandas as pd

from evaluate import load as evaluate_load
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from dataset.constants import WNCColumn
from dataset.preprocess import get_train_test_dataset
from utils import load_model, load_tokenizer

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
    _, test_dataset = get_train_test_dataset(tokenizer)
    subset_size = min(1000, len(test_dataset))
    test_dataset = test_dataset.select(range(subset_size))

    predictions = []
    references = []
    biased_list = []

    logger.info(f"Evaluating on {len(test_dataset)} examples...")

    batch_size = 16
    loader = DataLoader(test_dataset, batch_size=batch_size)

    for batch in tqdm(loader):
        biased_texts = batch[WNCColumn.BIASED]
        neutral_texts = batch[WNCColumn.NEUTRAL]

        inputs = tokenizer(
            [f"{b}\n" for b in biased_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        predicted_texts = [
            d[len(b) :].strip() if d.startswith(b) else d.strip()
            for d, b in zip(decoded, biased_texts)
        ]

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
