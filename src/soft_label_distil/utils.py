import json
from datetime import datetime
from logging import getLogger
from os import getenv
from pathlib import Path
from typing import List

import torch
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from transformers import PreTrainedTokenizer, TrainingArguments

from constants import DISTIL_OUTPUT_DIR, SYSTEM_PROMPT
from soft_label_distil.models import LogProbDTO, TopLogProbDTO

logger = getLogger(__name__)


CLIENT = OpenAI(
    api_key=getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1"
)

_TEACHER_CACHE: dict[str, List[LogProbDTO]] = {}


def get_distillation_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=DISTIL_OUTPUT_DIR.as_posix(),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.001,
        max_grad_norm=1.0,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="qwen3-distillation",
        dataloader_num_workers=2,
        fp16=True,
        lr_scheduler_type="cosine",
    )


def get_teacher_logprobs(biased_text: str, teacher_model_name: str) -> List[LogProbDTO]:
    if biased_text in _TEACHER_CACHE:
        return _TEACHER_CACHE[biased_text]

    try:
        response = CLIENT.chat.completions.create(
            model=teacher_model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Debias this sentence: {biased_text}",
                },
            ],
            max_completion_tokens=256,
            temperature=1.5,
            top_p=0.95,
            logprobs=True,
            top_logprobs=5,
        )

        output_file = DISTIL_OUTPUT_DIR / "teacher_responses.jsonl"
        save_teacher_response(biased_text, response, output_file)

        logprobs = extract_logprobs_from_response(biased_text, response)

        _TEACHER_CACHE[biased_text] = logprobs
        return logprobs

    except Exception as e:
        logger.error(f"Error getting teacher logprobs: {e}")
        return []


def extract_logprobs_from_response(
    biased_text: str, response: ChatCompletion
) -> List[LogProbDTO]:
    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        logger.error(
            f"No logprobs found in the response for biased text: {biased_text}, full response: {response}"
        )
        return []

    return [
        LogProbDTO(
            token_logprob.token,
            token_logprob.logprob,
            [
                TopLogProbDTO(top_logprob.token, top_logprob.logprob)
                for top_logprob in (token_logprob.top_logprobs or [])
            ],
        )
        for token_logprob in response.choices[0].logprobs.content
    ]


def build_teacher_distribution(
    teacher_token_logprob: LogProbDTO,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    teacher_probabilities = torch.zeros(vocab_size, device=device)
    raw_coverage = 0.0

    chosen_token_ids = tokenizer.encode(
        teacher_token_logprob.token, add_special_tokens=False
    )
    chosen_token_id = None

    if chosen_token_ids:
        chosen_token_id = chosen_token_ids[0]
        chosen_prob = torch.exp(
            torch.tensor(
                teacher_token_logprob.logprob / temperature,
                device=device,
            )
        )
        teacher_probabilities[chosen_token_id] = chosen_prob
        raw_coverage += chosen_prob.item()

    for top_logprob in teacher_token_logprob.top_logprobs:
        token_ids = tokenizer.encode(top_logprob.token, add_special_tokens=False)

        if not token_ids:
            continue

        token_id = token_ids[0]

        is_chosen_token_id = chosen_token_id is not None and token_id == chosen_token_id
        if is_chosen_token_id:
            continue

        prob = torch.exp(
            torch.tensor(
                top_logprob.logprob / temperature,
                device=device,
            )
        )
        teacher_probabilities[token_id] = prob
        raw_coverage += prob.item()

    return teacher_probabilities, raw_coverage


def save_teacher_response(
    biased_text: str, response: ChatCompletion, output_file: Path
) -> None:
    try:
        neutralized_text = (
            response.choices[0].message.content if response.choices else None
        )

        logprobs_data = []
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            for token_logprob in response.choices[0].logprobs.content:
                logprobs_data.append(
                    {
                        "token": token_logprob.token,
                        "logprob": token_logprob.logprob,
                        "top_logprobs": [
                            {"token": top.token, "logprob": top.logprob}
                            for top in (token_logprob.top_logprobs or [])
                        ],
                    }
                )

        record = {
            "timestamp": datetime.now().isoformat(),
            "biased_text": biased_text,
            "neutralized_text": neutralized_text,
            "logprobs": logprobs_data,
            "model": response.model if hasattr(response, "model") else None,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    except Exception as e:
        logger.error(f"Error saving teacher response: {e}")


def load_teacher_responses(input_file: Path) -> dict[str, List[LogProbDTO]]:
    responses = {}
    try:
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                biased_text = record.get("biased_text")
                logprobs_data = record.get("logprobs", [])

                if not biased_text or not logprobs_data:
                    continue

                logprobs = [
                    LogProbDTO(
                        token=item["token"],
                        logprob=item["logprob"],
                        top_logprobs=[
                            TopLogProbDTO(token=top["token"], logprob=top["logprob"])
                            for top in item.get("top_logprobs", [])
                        ],
                    )
                    for item in logprobs_data
                ]

                responses[biased_text] = logprobs

        logger.info(f"Loaded {len(responses)} teacher responses from {input_file}")

        _TEACHER_CACHE.update(responses)

    except FileNotFoundError:
        logger.warning(
            f"Teacher responses file not found: {input_file}, continuing with empty cache."
        )
    except Exception as e:
        logger.error(f"Error loading teacher responses: {e}")
    return responses
