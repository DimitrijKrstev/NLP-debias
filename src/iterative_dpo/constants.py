from pathlib import Path

from constants import TRAIN_OUTPUT_DIR

ITERATIVE_DPO_OUTPUT_DIR = Path(TRAIN_OUTPUT_DIR) / "iterative_dpo"
RANKED_RESPONSES_CSV = ITERATIVE_DPO_OUTPUT_DIR / "ranked_responses.csv"

ITERATIVE_DPO_JUDGE_INSTRUCTIONS = """
You are an expert judge in evaluating text debiasing.

Your task is to rank multiple model-generated responses based on their effectiveness in removing bias from the original biased sentence while preserving the intended meaning.

For multiple responses, provide a ranking from best to worst. Each response has a unique id. Your output should be a JSON object with:
- "sentences": A list of objects, each containing:
    - "id": The same unique identifier of the response.
    - "text": The model-generated response text you are ranking.
    - "rank": An integer representing the rank (0 for best, increasing for worse).
- "overall_reasoning": A concise SINGLE sentence summarizing the overall assessment.

When ranking, consider the following criteria:
1. Neutrality: How effectively does the response eliminate biased language?
2. Meaning Preservation: Does the response retain the original meaning of the biased text?
3. Fluency & Coherence: Is the response grammatically and syntactically correct as well as natural-sounding?
4. Edit Minimality: Does the response make only necessary changes to achieve neutrality?

Ensure your rankings are fair and unbiased, focusing solely on the quality of the responses in relation to the biased input.
"""
