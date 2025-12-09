from logging import getLogger

import mlflow
from trl.trainer.grpo_trainer import GRPOTrainer

from constants import RL_OUTPUT_DIR
from dataset.enums import DatasetSplit, TokenizationType
from dataset.preprocess import get_dataset_split
from rl.utils import get_grpo_config, reward_func, sync_model_tokenizer_config
from utils import load_peft_model, load_tokenizer

logger = getLogger(__name__)


def run_rlhf_training(model_name: str, mlflow_experiment: str, quantize: bool) -> None:
    mlflow.set_experiment(mlflow_experiment)

    tokenizer = load_tokenizer(model_name)
    model = load_peft_model(model_name, quantize)
    sync_model_tokenizer_config(model, tokenizer)

    dataset = get_dataset_split(DatasetSplit.TRAIN, TokenizationType.GRPO, None)

    model.train()
    grpo_config = get_grpo_config(model_name)
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    logger.info("Starting GRPO training with sentence-level judge scoring...")
    grpo_trainer.train()

    model.save_pretrained(RL_OUTPUT_DIR / "grpo-debiasing-model")
    tokenizer.save_pretrained(RL_OUTPUT_DIR / "grpo-debiasing-model")
    mlflow.log_artifact((RL_OUTPUT_DIR / "grpo-debiasing-model").as_posix())
