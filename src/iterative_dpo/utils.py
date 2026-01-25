from csv import DictWriter
from logging import getLogger
from os import getenv
from pathlib import Path
from typing import List

import mlflow
import torch
from dotenv import load_dotenv
from openai import OpenAI
from torch import no_grad
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.dpo_config import DPOConfig

from iterative_dpo.constants import (
    ITERATIVE_DPO_JUDGE_INSTRUCTIONS,
    RANKED_RESPONSES_CSV,
)
from iterative_dpo.models import (
    ModelOutputSentenceWithRank,
    ModelPreference,
    PreferencePair,
    SentenceWithRank,
)
from utils import remove_thinking_tags

load_dotenv()

OPENAI_KEY = getenv("OPENAI_API_KEY")
CLIENT = OpenAI(api_key=OPENAI_KEY)

logger = getLogger(__name__)


def get_dpo_config(model_name: str) -> DPOConfig:
    bf16_supported = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )

    return DPOConfig(
        output_dir=model_name,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=1e-6,
        beta=0.1,
        max_prompt_length=256,
        max_length=512,
        logging_steps=10,
        save_strategy="no",
        bf16=bf16_supported,
        fp16=not bf16_supported and torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="mlflow",
    )


def generate_responses_with_temperatures(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    formatted_prompt: str,
    temperatures: list[float] = [0.7, 1.0, 1.5],
) -> list[str]:
    responses = []

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_length = inputs["input_ids"].shape[1]  # type: ignore
    inputs = inputs.to(model.device)

    for temp in temperatures:
        with no_grad():
            outputs = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=512,
                temperature=temp,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        responses.append(remove_thinking_tags(response))

    return responses


def create_preference_pairs(
    prompts: list[list[dict[str, str]]],
    formatted_prompts: List[str],
    biased_texts: List[str],
    ground_truth_texts: List[str],
    all_responses: List[List[str]],
    judge_model_name: str,
    iteration: int = 0,
) -> list[PreferencePair]:
    pairs = []
    for sample_idx, (
        prompt,
        formatted_prompt,
        biased_text,
        ground_truth_text,
        responses,
    ) in enumerate(
        zip(prompts, formatted_prompts, biased_texts, ground_truth_texts, all_responses)
    ):
        sorted_responses = _rank_responses_with_judge(
            biased_text,
            [ground_truth_text] + responses,
            judge_model_name,
            iteration,
            sample_idx,
        )
        pairs = [
            PreferencePair(
                prompt,
                formatted_prompt,
                sorted_responses[better_idx],
                sorted_responses[worse_idx],
            )
            for better_idx, worse_idx in zip([0, 0, 1], [1, 3, 3])
        ] + [
            PreferencePair(
                prompt,
                formatted_prompt,
                sorted_responses[i],
                biased_text,
            )
            for i in range(3)
        ]

    return pairs


def _rank_responses_with_judge(
    biased_text: str,
    responses: list[str],
    judge_model_name: str,
    iteration: int = 0,
    sample_idx: int = 0,
) -> list[str]:
    sentences_with_ranks = [
        SentenceWithRank(i, i, response) for i, response in enumerate(responses)
    ]
    model_preference = _get_judge_rankings(
        biased_text,
        sentences_with_ranks,
        judge_model_name,
    )

    try:
        ground_truth = next(
            sentence
            for sentence in model_preference.sentences
            if sentence.id == sentences_with_ranks[0].id
        )
        logger.info(
            f"Rank for ground truth: {ground_truth.rank} "
            f"best rank id: {min(model_preference.sentences, key=lambda x: x.rank).id} "
            f"worst rank id: {max(model_preference.sentences, key=lambda x: x.rank).id}"
        )
    except StopIteration:
        logger.error(
            f"Ground truth response not found in judge rankings. for rankings: {sentences_with_ranks}"
        )

    id_to_rank = {sentence.id: sentence.rank for sentence in model_preference.sentences}

    response_with_sentence_text = [
        SentenceWithRank(
            original.id, id_to_rank.get(original.id, original.rank), original.text
        )
        for original in sentences_with_ranks
    ]

    _append_rankings_to_csv(
        response_with_sentence_text,
        sentences_with_ranks,
        biased_text,
        iteration,
        sample_idx,
    )

    ranked_responses = [
        response.text
        for response in sorted(response_with_sentence_text, key=lambda x: x.rank)
    ]
    logger.info(
        f"Ground truth is chosen as best: {ranked_responses[0] == responses[0]}"
    )

    ranking_comparison = {
        "biased_text": biased_text,
        "original_list": [
            {"id": s.id, "text": s.text, "original_rank": s.rank}
            for s in sentences_with_ranks
        ],
        "judge_sorted_list": [
            {"id": s.id, "judge_rank": s.rank}
            for s in sorted(
                model_preference.sentences, key=lambda x: x.rank, reverse=True
            )
        ],
        "reasoning": model_preference.overall_reasoning,
    }
    mlflow.log_dict(
        ranking_comparison, f"rankings/iter_{iteration}_sample_{sample_idx}.json"
    )

    return ranked_responses


def _get_judge_rankings(
    biased_text: str,
    responses: list[SentenceWithRank],
    judge_model_name: str,
) -> ModelPreference:
    try:
        ranks = CLIENT.chat.completions.parse(
            model=judge_model_name,
            response_format=ModelPreference,
            messages=[
                {"role": "system", "content": ITERATIVE_DPO_JUDGE_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": _get_judge_user_prompt(biased_text, responses),
                },
            ],
            max_completion_tokens=2048,
        )
        if ranks.choices[0].message.parsed is None:
            raise ValueError(f"Output not received from {judge_model_name}")
    except Exception as e:
        logger.error(
            f"Error getting response from {judge_model_name}: {e}, fallbacking to status quo ranking."
        )
        fallback_sentences = [
            ModelOutputSentenceWithRank(id=s.id, rank=s.rank) for s in responses
        ]
        return ModelPreference(sentences=fallback_sentences, overall_reasoning=None)

    return ranks.choices[0].message.parsed


def _get_judge_user_prompt(biased_text: str, responses: list[SentenceWithRank]) -> str:
    return f"""
    The following is a biased text sample from a research dataset used to train bias detection models:

    Original biased text: "{biased_text}"

    Task: Rank the following debiased responses from best (0) to worst (N) based on how well they remove bias while preserving meaning.

    Responses to evaluate:
    {responses}"""


def _append_rankings_to_csv(
    response_with_sentence_text: list[SentenceWithRank],
    original_ranks: list[SentenceWithRank],
    biased_text: str,
    iteration: int,
    sample_idx: int,
) -> None:
    csv_path = Path(RANKED_RESPONSES_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = [
        "iteration",
        "sample_idx",
        "biased_text",
        "response_id",
        "original_rank",
        "judge_rank",
        "response_text",
    ]

    id_to_original_rank = {s.id: s.rank for s in original_ranks}

    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for response in response_with_sentence_text:
            writer.writerow(
                {
                    "iteration": iteration,
                    "sample_idx": sample_idx,
                    "biased_text": biased_text,
                    "response_id": response.id,
                    "original_rank": id_to_original_rank.get(response.id, -1),
                    "judge_rank": response.rank,
                    "response_text": response.text,
                }
            )
