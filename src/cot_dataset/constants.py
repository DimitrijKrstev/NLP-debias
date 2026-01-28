from pathlib import Path

OUTPUT_DIR = Path("cot_dataset_output/")
COT_DATASET_FILE = OUTPUT_DIR / "cot_dataset.jsonl"

# Model to use for generation
TEACHER_MODEL = "openrouter/openai/gpt-oss-120b"

# Concurrency settings
MAX_CONCURRENT_REQUESTS = 20
SAVE_EVERY = 50

# Target style for debiasing
TARGET_STYLE = "Neutral, unbiased"
