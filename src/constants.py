from pathlib import Path

SFT_RL_CUTOFF = 0.9

TRAIN_OUTPUT_DIR = "./output"

RL_OUTPUT_DIR = Path("judge/output")
RL_CACHE_FILE = Path("judge/judge_cache.jsonl")
JUDGE_SCORE_FILE = RL_OUTPUT_DIR / "judge_scores.csv"
