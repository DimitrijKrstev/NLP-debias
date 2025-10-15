import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import openai
from trl import (
    PPOTrainer,
    PPOConfig,
)
from transformers import set_seed, AutoModelForCausalLM, GenerationConfig
import requests
from constants import JUDGE_OUTPUT_DIR
from dataset.preprocess import get_train_test_dataset
from judge.prompt import build_judge_prompt
from utils import (
    get_model_and_tokenizer,
    load_cache,
    make_cache_key,
    save_to_cache,
)
from typing import Tuple
from datasets import Dataset
import torch.nn as nn


class DummyRewardModel(nn.Module):
    """Dummy reward model that does nothing - we use custom reward function instead"""

    def __init__(self, device="cpu"):
        super().__init__()
        self.device_type = device
        # Add a dummy parameter so it's a valid nn.Module
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Return dummy rewards (we'll override these with our custom function)
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, device=input_ids.device)

    def to(self, device):
        self.device_type = device
        return super().to(device)

    def eval(self):
        return super().eval()


cache = load_cache()


def ollama_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"[Ollama error] {e}")
        return ""


def evaluate_sample_judge(biased_text, model_output, neutral_text, llm_model="llama3"):
    key = make_cache_key(biased_text, model_output, neutral_text)

    if key in cache:
        return cache[key]

    prompt = build_judge_prompt(biased_text, model_output, neutral_text)

    # response = openai.ChatCompletion.create(
    #     model=llm_model,
    #     messages=[
    #         {"role": "system", "content": "You are a precise and objective evaluator."},
    #         {"role": "user", "content": prompt},
    #     ],
    #     temperature=0.0,
    # )

    response_text = ollama_chat(
        model=llm_model,
        system_prompt="You are a precise and objective evaluator. Respond only in JSON.",
        user_prompt=prompt,
    )

    # content = response["choices"][0]["message"]["content"]

    # try:
    #     parsed = json.loads(content)
    # except json.JSONDecodeError:
    #     parsed = {"error": "invalid json", "raw": content}

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = {"error": "invalid json", "raw": response_text}

    parsed.update(
        {
            "biased_text": biased_text,
            "neutral_text": neutral_text,
            "model_output": model_output,
            "key": key,
        }
    )

    cache[key] = parsed
    save_to_cache(parsed)

    return parsed


def reward_function(
    biased: str, model_output: str, neutral_ref: str, llm_model: str
) -> float:

    parsed = evaluate_sample_judge(biased, model_output, neutral_ref, llm_model)

    reward = (
        parsed.get("neutrality", 0) * 0.4
        + parsed.get("meaning_preservation", 0) * 0.3
        + parsed.get("fluency", 0) * 0.15
        + parsed.get("faithfulness_to_reference", 0) * 0.1
        + parsed.get("edit_minimality", 0) * 0.05
    )

    reward = np.clip(reward, 0.0, 1.0)
    return reward


def run_rlhf_training(
    model_tokenizer_path: str, llm_model: str = "llama3"
) -> Tuple[object, object]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    model, tokenizer = get_model_and_tokenizer(model_tokenizer_path, True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not hasattr(model, "generation_config") or model.generation_config is None:
        model.generation_config = GenerationConfig(
            eos_token_id=(
                model.config.eos_token_id
                if hasattr(model.config, "eos_token_id")
                else tokenizer.eos_token_id
            ),
            pad_token_id=tokenizer.pad_token_id,
        )

    model.to(device)
    model.train()

    train_dataset, _ = get_train_test_dataset(tokenizer)
    dataset_list = []
    for item in train_dataset:
        query_text = item.get("biased", "")
        neutral_text = item.get("neutral", "")

        tokenized = tokenizer(
            query_text,
            truncation=True,
            max_length=512,
        )

        dataset_list.append(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "query_text": query_text,
                "neutral_text": neutral_text,
            }
        )

    dataset = Dataset.from_list(dataset_list)

    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=2,
        num_ppo_epochs=4,
        vf_coef=0.0,
        kl_coef=0.0,
    )

    reward_model = DummyRewardModel(device=device)
    reward_model.to(device)
    reward_model.eval()

    ref_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m",
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )
    ref_model.to("cpu")

    value_model = model
    value_model.base_model_prefix = "model"

    from transformers import DataCollatorWithPadding

    class CustomDataCollator(DataCollatorWithPadding):
        def __call__(self, features):
            # Separate text fields from tokenized fields
            text_fields = {}
            tokenized_features = []

            for feature in features:
                text_fields.setdefault("query_text", []).append(
                    feature.pop("query_text")
                )
                text_fields.setdefault("neutral_text", []).append(
                    feature.pop("neutral_text")
                )
                tokenized_features.append(feature)

            # Pad tokenized fields
            batch = super().__call__(tokenized_features)

            # Add text fields back (as lists, not tensors)
            batch.update(text_fields)

            return batch

    data_collator = CustomDataCollator(tokenizer=tokenizer, padding=True)

    trainer = PPOTrainer(
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=ppo_config,
        train_dataset=dataset,
        value_model=value_model,
        data_collator=data_collator,
    )

    print("[INFO] Starting PPO RLHF loop...")

    for epoch in range(max(1, int(getattr(ppo_config, "ppo_epochs", 1)))):
        epoch_rewards = []

        for batch in tqdm(trainer.dataloader, desc=f"Epoch {epoch+1}"):
            queries = batch["query_text"]
            neutrals = batch["neutral_text"]

            response_tensors = trainer.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
            )
            responses = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )

            rewards = []

            for query, response, neutral in zip(queries, responses, neutrals):
                reward = reward_function(query, response, neutral, llm_model)
                rewards.append(reward)
                epoch_rewards.append(reward)

            trainer.step(
                batch["input_ids"],
                response_tensors,
                rewards,
            )
        avg_reward = np.mean(epoch_rewards) if len(epoch_rewards) > 0 else 0.0
        print(f"[INFO] Completed epoch {epoch+1} — avg reward: {avg_reward:.4f}")

    print("[INFO] PPO RLHF training finished.")
    return model, tokenizer
