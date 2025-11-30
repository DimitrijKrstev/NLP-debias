from logging import getLogger
import mlflow

from trl.trainer.grpo_trainer import GRPOTrainer
from trl import DPOTrainer

from constants import RL_OUTPUT_DIR
from dataset.preprocess import get_train_dataset
from utils import load_peft_model, load_tokenizer
from rl.utils import (
    get_grpo_config,
    reward_func,
    sync_model_tokenizer_config,
)

logger = getLogger(__name__)


def judge_and_make_pairs(grpo_batch, judge_fn):
    pairs = []

    for item in grpo_batch:
        prompt = item["prompt"]
        samples = item["samples"]

        best, worst = judge_fn(prompt, samples)

        pairs.append({
            "prompt": prompt,
            "chosen": best,
            "rejected": worst,
        })

    return pairs


def judge_fn(prompt, samples):
    scores = reward_func(prompt, samples)
    best = samples[scores.index(max(scores))]
    worst = samples[scores.index(min(scores))]
    return best, worst

def run_hybrid_training(
    model_name: str,
    mlflow_experiment: str,
    quantize: bool,
    cycles: int = 3,
    grpo_steps_per_cycle: int = 50,
    dpo_epochs: int = 1,
):
    mlflow.set_experiment(mlflow_experiment)

    logger.info("Loading model + tokenizer...")
    tokenizer = load_tokenizer(model_name)
    model = load_peft_model(model_name, quantize)
    sync_model_tokenizer_config(model, tokenizer)

    dataset = get_train_dataset(tokenizer, with_prompts_only=True)

    grpo_config = get_grpo_config(model_name)

    grpo = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
    )

    for cycle in range(cycles):
        logger.info(f"\n===== Hybrid Cycle {cycle+1}/{cycles} =====")

        logger.info("Sampling on-policy with GRPO...")
        grpo_batch = grpo.generate_samples(
            steps=grpo_steps_per_cycle,
            return_full_data=True,
        )

        logger.info("Judging samples to construct DPO dataset...")
        dpo_pairs = judge_and_make_pairs(grpo_batch, judge_fn)

        logger.info("Running DPO fine-tuning on preference pairs...")
        dpo_trainer = DPOTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dpo_pairs,
            args=dict(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                num_train_epochs=dpo_epochs,
                learning_rate=5e-6,
                remove_unused_columns=False,
                output_dir=str(RL_OUTPUT_DIR / f"hybrid-cycle-{cycle}"),
            ),
        )

        dpo_trainer.train()

    logger.info("Saving final hybrid model...")
    out = RL_OUTPUT_DIR / "hybrid-grpo-dpo"
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    mlflow.log_artifact(out.as_posix())
