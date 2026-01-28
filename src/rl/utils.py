from logging import getLogger

from trl.trainer.grpo_config import GRPOConfig

from constants import JUDGE_SCORE_FILE
from dataset.enums import WNCColumn
from judge.main import get_judge_score
from utils import remove_thinking_tags

logger = getLogger(__name__)


def reward_func(
    prompts: list[str],
    completions: list[dict],
    **kwargs,
) -> list[float]:
    rewards = []

    biased_texts = kwargs[WNCColumn.BIASED]
    neutral_texts = kwargs[WNCColumn.NEUTRAL]

    for completion, biased_text, neutral_text in zip(
        completions, biased_texts, neutral_texts
    ):
        model_output = remove_thinking_tags(completion[0]["content"])
        score = get_judge_score(
            biased_text, model_output, neutral_text, "gpt-5-mini", JUDGE_SCORE_FILE
        )

        rewards.append(score)

    return rewards


def sync_model_tokenizer_config(model, tokenizer):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


def get_grpo_config(model_name: str, output_dir: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=output_dir,
        run_name=f"{model_name}-grpo-debiasing",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-6,
        num_generations=3,
        generation_batch_size=3,
        max_prompt_length=256,
        max_completion_length=256,
        beta=0.01,
        epsilon=0.2,
        loss_type="dapo",
        temperature=0.7,
        top_p=0.9,
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=False,
        bf16=False,
        remove_unused_columns=False,
        report_to="mlflow",
    )
