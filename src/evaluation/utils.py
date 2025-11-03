import os
import re

import torch
from evaluate import load as evaluate_load  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer, util

from evaluation.models import Metrics

os.environ["MPLBACKEND"] = "Agg"


def compute_metrics(predictions: list[str], references: list[str]) -> Metrics:
    bleu = evaluate_load("bleu")
    meteor = evaluate_load("meteor")
    rouge = evaluate_load("rouge")
    perplexity = evaluate_load("perplexity", module_type="metric")
    bertscore = evaluate_load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    perplexity_score = perplexity.compute(
        predictions=predictions, references=references
    )
    bertscore_result = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-uncased",  
    )

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = similarity_model.encode(predictions, convert_to_tensor=True)
    ref_emb = similarity_model.encode(references, convert_to_tensor=True)
    semantic_sim = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()
    bertscore_f1 = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

    return Metrics.from_scores(
        bleu_score=bleu_score,
        meteor_score=meteor_score,
        rouge_scores=rouge_scores,
        perplexity_score=perplexity_score,
        semantic_sim=semantic_sim,
        bertscore_f1=bertscore_f1,
    )


def clean_output(text: str) -> str:
    text = text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0]
    text = re.sub(r"(</?think>|<\|.*?\|>)", "", text)
    return text.strip().strip('"').strip("'").replace("\n\n", "\n")


def make_chat_prompt(biased_text: str) -> str:
    return (
        f"<|im_start|>system\n"
        "You are an expert at minimal text debiasing. "
        "Your goal is to make only the smallest necessary edits to make the text neutral and objective, "
        "while preserving the original meaning and wording as much as possible.<|im_end|>\n"
        f"<|im_start|>user\n"
        "Rewrite this biased text to be neutral. "
        "Perform minimal changes — only adjust words or phrasing that introduce bias. "
        "Do not explain or add new content.\n"
        f"Biased text: {biased_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def debias_text(text: list[str], model, tokenizer, max_length: int = 256):
    inputs = tokenizer(
        [f"{b}\n" for b in text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    predicted_texts = [d.strip() for d in decoded_outputs]

    return predicted_texts
