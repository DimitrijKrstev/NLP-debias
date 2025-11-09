import csv
import re
import time
from logging import getLogger
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from trl.trainer.grpo_config import GRPOConfig

from constants import JUDGE_SCORE_FILE
from dataset.constants import WNCColumn
from rl.models import ModelResponseEvaluation
from rl.prompt import build_judge_prompt, get_judge_instructions

logger = getLogger(__name__)

load_dotenv()
OPENAI_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def get_judge_score(
    biased_text: str,
    model_output: str,
    reference_text: str,
    openai_model: str = "gpt-5-mini",
    max_retries: int = 3,
) -> float:

    instructions = get_judge_instructions()
    prompt = build_judge_prompt(biased_text, model_output, reference_text)

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=openai_model,
                instructions=instructions,
                input=prompt,
                text_format=ModelResponseEvaluation,
            )
            score = response.output_parsed

            if not score:
                logger.error(f"Empty response from {openai_model}")
                return 0.0

            total_score = score.get_normalized_full_score()

            logger.info(
                f"Judge score: {total_score:.2f} | "
                f"Neautrality:{score.neutrality} Meaning Perservation:{score.meaning_preservation} "
                f"Fluency:{score.fluency}"
            )
            append_results_to_csv(
                biased_text, model_output, reference_text, total_score
            )

            return total_score

        except Exception as e:
            logger.warning(f"Judge attempt {attempt + 1}/{max_retries} failed: {e}")

            if attempt < max_retries - 1:
                sleep_time = 2**attempt
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(
                    f"All judge attempts failed for output: {model_output[:100]}...",
                    exc_info=True,
                )

    return 0.0


def append_results_to_csv(
    biased_text: str,
    model_output: str,
    reference_text: str,
    score: float | None,
) -> None:
    path = Path(JUDGE_SCORE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()

    with open(JUDGE_SCORE_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["biased_text", "model_output", "reference_text", "score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "biased_text": biased_text,
                "model_output": model_output,
                "reference_text": reference_text,
                "score": score,
            }
        )


def reward_func(
    prompts: list[str],
    completions: list[dict],
    **kwargs,
) -> list[float]:
    rewards = []

    biased_texts = kwargs[WNCColumn.BIASED]
    neutral_texts = kwargs[WNCColumn.NEUTRAL]

    for completion, biased_text, neutral_text in zip(
        completions, biased_texts, neutral_texts
    ):
        model_output = remove_thinking_tags(completion[0]["content"])
        score = get_judge_score(
            biased_text=biased_text,
            model_output=model_output,
            reference_text=neutral_text,
        )

        rewards.append(score if score is not None else 0.0)

    return rewards


def sync_model_tokenizer_config(model, tokenizer):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def get_grpo_config(model_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir="./grpo-debiasing-model",
        run_name=f"{model_name}-grpo-debiasing",
        per_device_train_batch_size=3,        
        num_train_epochs=3,
        learning_rate=1e-6,
        num_generations=3,
        generation_batch_size=3,
        max_prompt_length=256,
        max_completion_length=256,
        beta=0.01,
        epsilon=0.2,
        loss_type="dapo",
        temperature=0.7,
        top_p=0.9,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,                
        save_total_limit=3,
        gradient_checkpointing=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="mlflow",
    )


def remove_thinking_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
