from pathlib import Path

SFT_RL_CUTOFF = 0.9

TRAIN_OUTPUT_DIR = "./output"

RL_OUTPUT_DIR = Path("rl_output/")
RL_CACHE_FILE = RL_OUTPUT_DIR / "judge_cache.jsonl"
JUDGE_SCORE_FILE = RL_OUTPUT_DIR / "judge_scores.csv"

GRPO_SYSTEM_PROMPT = (
    "You are an expert debiaser specializing in creating a minimal neutral version of biased text "
    "while preserving its original meaning. For a given biased sentence input, return ONLY its short and minimal unbiased "
    "version and nothing else"
)
