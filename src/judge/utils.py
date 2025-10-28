from logging import getLogger
from dotenv import load_dotenv
from openai import OpenAI
from judge.models import ModelResponseEvaluation
from os import getenv

from judge.prompt import build_judge_prompt, get_judge_instructions

logger = getLogger(__name__)

load_dotenv()
OPENAI_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def get_judge_score(
    biased_text: str,
    model_output: str,
    reference_text: str,
    openai_model: str = "gpt-4o-mini",
) -> float | None:
    instructions = get_judge_instructions()
    prompt = build_judge_prompt(biased_text, model_output, reference_text)

    try:
        response = client.responses.parse(
            model=openai_model,
            instructions=instructions,
            input=prompt,
            text_format=ModelResponseEvaluation,
        )
        score = response.output_parsed
        logger.info(score)
        if not score:
            logger.error(f"Empty response from {openai_model}")
            return None

        return score.sum_all()

    except Exception as e:
        logger.error(f"OpenAI Judge error {e}")
        return None


def split_into_sentences(text):
    import re

    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]
