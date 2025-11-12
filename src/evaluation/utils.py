import os

import torch
from evaluate import load as evaluate_load  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer, util

from constants import GRPO_SYSTEM_PROMPT
from evaluation.models import Metrics

os.environ["MPLBACKEND"] = "Agg"


def compute_metrics(predictions: list[str], references: list[str]) -> Metrics:
    bleu = evaluate_load("bleu")
    meteor = evaluate_load("meteor")
    rouge = evaluate_load("rouge")
    bertscore = evaluate_load("bertscore")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
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
        semantic_sim=semantic_sim,
        bertscore_f1=bertscore_f1,
    )


def clean_output(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]

    return text


def make_chat_prompt(biased_text: str) -> list[dict]:
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": f"Make this neutral: {biased_text}"},
    ]


def debias_text(texts: list[str], model, tokenizer, max_length: int = 256) -> list[str]:
    chat_prompts = [make_chat_prompt(text) for text in texts]

    formatted_prompts = tokenizer.apply_chat_template(
        chat_prompts,
        tokenize=False,
        add_generation_prompt=False,
    )

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

    decoded_outputs = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=False
    )

    predicted_texts = [clean_output(text) for text in decoded_outputs]

    return predicted_texts
