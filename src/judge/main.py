from logging import getLogger

from constants import JUDGE_OUTPUT_DIR
from judge.constants import JUDGE_INSTRUCTIONS
from judge.models import ModelResponseEvaluation
from judge.utils import (
    append_results_to_csv,
    build_judge_prompt,
    get_instructor_client,
)

logger = getLogger(__name__)


CLIENT = get_instructor_client()


def get_judge_score(
    biased_text: str,
    model_output: str,
    reference_text: str,
    model: str,
    score_file_name: str | None,
) -> float:
    prompt = build_judge_prompt(biased_text, model_output, reference_text)

    try:
        score = CLIENT.chat.completions.create(
            model=model,
            response_model=ModelResponseEvaluation,
            messages=[
                {"role": "system", "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
    except Exception as e:
        logger.error(f"Error getting response from {model}: {e}")
        return 0.0

    if not score:
        logger.error(f"Empty response from {model}")
        return 0.0

    total_score = score.get_normalized_full_score()

    logger.info(
        f"Judge score: {total_score:.2f} | "
        f"Neutrality:{score.neutrality} Meaning Preservation:{score.meaning_preservation} "
        f"Fluency:{score.fluency}"
    )

    if score_file_name:
        output_file_path = (
            JUDGE_OUTPUT_DIR
            / f"{score_file_name.replace(".", "").replace("/", "_").replace(" ", "_")}.csv"
        )
        append_results_to_csv(
            biased_text, model_output, reference_text, score, output_file_path
        )

    return total_score
