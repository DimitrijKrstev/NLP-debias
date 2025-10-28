from logging import getLogger
import mlflow
from peft import PeftModel
from transformers import DataCollatorWithPadding
from trl.trainer.ppo_trainer import PPOTrainer
from trl.trainer.ppo_config import PPOConfig
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from dataset.preprocess import get_train_dataset
from judge.llm_judge import LLMJudgeRewardModel
from utils import load_model, load_tokenizer
from judge.utils import get_judge_score

logger = getLogger(__name__)


def run_rlhf_training(
    model_name: str,
    training_model_and_tokenizer: str,
    open_ai_remote_model_name: str,
    mlflow_experiment: str,
    quantize: bool,
) -> None:
    mlflow.set_experiment(mlflow_experiment)
    mlflow.start_run(run_name=f"{training_model_and_tokenizer}-rlhf-debiasing")

    base_model = load_model(model_name, quantize)
    tokenizer = load_tokenizer(training_model_and_tokenizer)

    model = PeftModel.from_pretrained(base_model, training_model_and_tokenizer)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    reference_base = load_model(model_name, quantize)
    reference_model = PeftModel.from_pretrained(
        reference_base, training_model_and_tokenizer
    )
    reference_model.eval()

    value_model = load_model(model_name, quantize)

    dataset = get_train_dataset(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    reward_model = LLMJudgeRewardModel(
        get_judge_score, open_ai_remote_model_name, tokenizer, dataset
    )

    ppo_config = PPOConfig(
        exp_name=f"{training_model_and_tokenizer}-rlhf",
        num_ppo_epochs=4,
        kl_coef=0.05,
        cliprange=0.2,
        vf_coef=0.1,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
        whiten_rewards=False,
    )

    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=reference_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # gen_kwargs = {
    #     "max_new_tokens": 128,
    #     "do_sample": True,
    #     "top_p": 0.9,
    #     "temperature": 0.7,
    #     "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    # }

    logger.info("Starting PPO training with sentence-level judge scoring...")
    ppo_trainer.train()

    # for epoch, batch in enumerate(ppo_trainer.dataloader):
    #     query_tensors = batch["input_ids"]
    #     biased_texts = batch.get("biased_text", [""] * len(query_tensors))
    #     reference_texts = batch.get("reference_text", [""] * len(query_tensors))

    #     reward_model.set_batch_context(biased_texts, reference_texts)

    #     with torch.no_grad():
    #         response_tensors = model.generate(
    #             query_tensors,
    #             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    #             **gen_kwargs
    #         )
    #         response_tensors = response_tensors[:, query_tensors.shape[1]:]

    #     rewards = reward_model(response_tensors).squeeze()

    #     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    #     if epoch % 10 == 0:
    #         if len(rewards) > 0:
    #             mlflow.log_metrics(
    #                 {
    #                     "reward_mean": rewards.mean().item(),
    #                     "reward_std": rewards.std().item(),
    #                 },
    #                 step=epoch,
    #             )
    #             logger.info(
    #                 f"Epoch {epoch} - Avg Reward: {rewards.mean():.3f}"
    #             )

    model.save_pretrained("./rlhf-debiasing-model")
    tokenizer.save_pretrained("./rlhf-debiasing-model")

    mlflow.log_artifact("./rlhf-debiasing-model")
    mlflow.end_run()
