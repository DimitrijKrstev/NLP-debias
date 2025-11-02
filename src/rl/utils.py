import csv
import time
from logging import getLogger
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from constants import JUDGE_SCORE_FILE
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
    openai_model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> float | None:

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
                return None

            total_score = score.sum_all()

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

    return None


def append_results_to_csv(
    biased_text: str,
    model_output: str,
    reference_text: str,
    score: float | None,
) -> None:
    file_exists = Path(JUDGE_SCORE_FILE).is_file()
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
