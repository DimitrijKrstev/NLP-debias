from pathlib import Path
from constants import RL_JUDGE_SCORE_FILE
from judge.utils import CLIENT, append_results_to_csv, build_judge_prompt
from judge.constants import JUDGE_INSTRUCTIONS
from judge.models import ModelResponseEvaluation
from rl.utils import logger


def get_judge_score(
    biased_text: str,
    model_output: str,
    reference_text: str,
    model: str = "gpt-5-mini",
    score_path: Path | None = RL_JUDGE_SCORE_FILE,
) -> float:
    instructions = JUDGE_INSTRUCTIONS
    prompt = build_judge_prompt(biased_text, model_output, reference_text)

    try:
        score = CLIENT.chat.completions.create(
            model=model,
            response_model=ModelResponseEvaluation,
            messages=[
                {"role": "system", "content": instructions},
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

    if score_path:
        append_results_to_csv(
            biased_text, model_output, reference_text, total_score, score_path
        )

    return total_score
