from csv import DictWriter
from logging import getLogger
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from instructor import from_openai  # type: ignore
from openai import OpenAI

from judge.models import ModelResponseEvaluation

logger = getLogger(__name__)


load_dotenv()

OPENAI_KEY = getenv("OPENAI_API_KEY")
OPENROUTER_KEY = getenv("OPENROUTER_API_KEY")


def get_instructor_client():
    if OPENROUTER_KEY:
        return from_openai(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_KEY,
            )
        )

    if not OPENAI_KEY:
        raise ValueError(
            "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is specified in environment variables. "
            "Provide one for Judge API access. OPENROUTER takes precedence."
        )
    return from_openai(OpenAI(api_key=OPENAI_KEY))


def build_judge_prompt(biased_text: str, model_output: str, reference_text: str) -> str:
    return f"""
        BIAS INPUT:
        {biased_text}

        MODEL OUTPUT:
        {model_output}

        REFERENCE (human neutral text):
        {reference_text}
        """


def append_results_to_csv(
    biased_text: str,
    model_output: str,
    reference_text: str,
    score: ModelResponseEvaluation,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.is_file()

    with open(path, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model_output",
            "neutrality",
            "meaning_preservation",
            "fluency",
            "edit_minimality",
            "total_score",
            "biased_text",
            "reference_text",
            "overall_reasoning",
        ]
        writer = DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "model_output": model_output,
                "neutrality": score.neutrality,
                "meaning_preservation": score.meaning_preservation,
                "fluency": score.fluency,
                "edit_minimality": score.edit_minimality,
                "total_score": score.get_normalized_full_score(),
                "biased_text": biased_text,
                "reference_text": reference_text,
                "overall_reasoning": score.overall_reasoning,
            }
        )
