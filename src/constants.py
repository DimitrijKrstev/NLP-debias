from pathlib import Path

SFT_RL_CUTOFF = 0.9

TRAIN_OUTPUT_DIR = Path("./output")

RL_OUTPUT_DIR = Path("rl_output/")
RL_CACHE_FILE = RL_OUTPUT_DIR / "judge_cache.jsonl"
JUDGE_SCORE_FILE = RL_OUTPUT_DIR / "judge_scores.csv"

GRPO_SYSTEM_PROMPT = """You are an expert text debiaser. Your task is to transform biased sentences into neutral, unbiased versions.

CRITICAL INSTRUCTIONS:
- Output ONLY the unbiased sentence
- Do NOT include any thinking process, reasoning, or explanations
- Do NOT use tags like <think>, <reasoning>, or similar
- Do NOT write "Here's the unbiased version:" or any preamble
- Do NOT add commentary, notes, or justifications
- Start your response directly with the neutral sentence

REMEMBER: Your entire response should be ONLY the neutral sentence. Begin your response immediately with the unbiased version."""
