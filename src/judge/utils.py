from functools import lru_cache
from logging import getLogger
import time
from dotenv import load_dotenv
from openai import OpenAI
from judge.models import ModelResponseEvaluation
from os import getenv
import nltk
import hashlib

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import sent_tokenize
from judge.prompt import build_judge_prompt, get_judge_instructions

logger = getLogger(__name__)

load_dotenv()
OPENAI_KEY = getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1000)
def _get_judge_score_cached(
    biased_text_hash: str,
    model_output_hash: str,
    reference_text_hash: str,
    openai_model: str,
) -> float | None:
    return get_judge_score(
        biased_text=biased_text_hash,
        model_output=model_output_hash,
        reference_text=reference_text_hash,
        openai_model=openai_model,
        use_cache=False,
    )


def get_judge_score(
    biased_text: str,
    model_output: str,
    reference_text: str,
    openai_model: str = "gpt-4o-mini",
    max_retries: int = 3,
    use_cache: bool = True,
) -> float | None:

    if use_cache:
        return _get_judge_score_cached(
            _hash_text(biased_text),
            _hash_text(model_output),
            _hash_text(reference_text),
            openai_model,
        )

    else:
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
                    f"N:{score.neutrality} M:{score.meaning_preservation} "
                    f"F:{score.fluency} | {score.overall_reasoning}"
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

    return None


def split_into_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text.strip().replace("\n", " "))
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 5]

    return cleaned or [text.strip()]
