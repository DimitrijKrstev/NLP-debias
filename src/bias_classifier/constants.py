from pathlib import Path

MODEL_NAME = "microsoft/deberta-v3-base"

OUTPUT_DIR = Path("bias_classifier_output/")
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"

MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
EVAL_STEPS: int | None = 40  # None = eval per epoch, int = eval every N steps
MAX_TRAIN_SAMPLES: int | None = None  # None = use all samples in slice, int = cap at N WNC pairs

LABEL_NEUTRAL = 0
LABEL_BIASED = 1

ID2LABEL = {LABEL_NEUTRAL: "neutral", LABEL_BIASED: "biased"}
LABEL2ID = {"neutral": LABEL_NEUTRAL, "biased": LABEL_BIASED}
