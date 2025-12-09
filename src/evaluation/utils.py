import os
from logging import getLogger

import torch
from evaluate import load as evaluate_load  # type: ignore[import-untyped]
from numpy import average
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from constants import GRPO_SYSTEM_PROMPT
from evaluation.models import Metrics
from judge.main import get_judge_score

os.environ["MPLBACKEND"] = "Agg"

logger = getLogger(__name__)


def compute_metrics(
    biased_references: list[str],
    references: list[str],
    predictions: list[str],
    judge_model_name: str,
    model_name: str,
) -> Metrics:
    bleu = evaluate_load("bleu")
    meteor = evaluate_load("meteor")
    rouge = evaluate_load("rouge")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bertscore_f1 = _compute_bertscore_f1(predictions, references)
    semantic_sim = _compute_semantic_similarity(predictions, references)
    judge_score = _compute_judge_score(
        biased_references, predictions, references, judge_model_name, model_name
    )

    return Metrics.from_scores(
        bleu_score=bleu_score,
        meteor_score=meteor_score,
        rouge_scores=rouge_scores,
        semantic_sim=semantic_sim,
        bertscore_f1=bertscore_f1,
        judge_score=judge_score,
    )


def _compute_bertscore_f1(predictions: list[str], references: list[str]) -> float:
    bertscore = evaluate_load("bertscore")
    bertscore_result = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-uncased",
    )

    if bertscore_result and bertscore_result.get("f1"):
        return sum(bertscore_result["f1"]) / len(bertscore_result["f1"])

    logger.warning(
        f"BERTScore F1 cannot be computed for predictions and references. \n\n"
        f"Predictions: \n{predictions}, \n\nReferences: \n{references}"
    )
    return 0.0


def _compute_semantic_similarity(
    predictions: list[str], references: list[str]
) -> float:
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = similarity_model.encode(predictions, convert_to_tensor=True)
    ref_emb = similarity_model.encode(references, convert_to_tensor=True)

    return util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()


def _compute_judge_score(
    biased_references: list[str],
    predictions: list[str],
    references: list[str],
    judge_model_name: str,
    model_name: str,
) -> float:
    judge_scores = [
        get_judge_score(biased, predicted, reference, judge_model_name, model_name)
        for biased, predicted, reference in tqdm(
            zip(biased_references, predictions, references),
            desc="Computing judge scores",
            total=len(predictions),
        )
    ]
    return average(judge_scores).item()


def debias_text(texts: list[str], model, tokenizer, max_length: int = 256) -> list[str]:
    chat_prompts = [_make_chat_prompt(text) for text in texts]

    formatted_prompts = tokenizer.apply_chat_template(
        chat_prompts,
        tokenize=False,
        add_generation_prompt=True,
    )

    logger.info(f"Example formatted prompt: {formatted_prompts[0] if formatted_prompts else 'None'}")

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_lengths = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_lengths:]

    decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    logger.info(f"Example raw output: '{decoded_outputs[0] if decoded_outputs else 'None'}'")

    predicted_texts = [_clean_output(text) for text in decoded_outputs]

    logger.info(f"Example cleaned output: '{predicted_texts[0] if predicted_texts else 'None'}'")

    return predicted_texts


def _make_chat_prompt(biased_text: str) -> list[dict]:
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": f"Make this neutral: {biased_text}"},
    ]


def _clean_output(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]

    return text
