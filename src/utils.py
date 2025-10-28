import json
import os
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from constants import JUDGE_CACHE_FILE, TRAIN_OUTPUT_DIR

HF_TOKEN = os.getenv("HF_TOKEN")


logger = getLogger(__name__)


def load_peft_model_and_tokenizer(model_name: str, quantize: bool) -> tuple[Any, Any]:
    logger.info(f"Loading model: {model_name}")

    model = load_model(model_name, quantize)
    tokenizer = load_tokenizer(model_name)

    if quantize:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_model(model_name: str | Path, quantize: bool) -> Any:
    bnb_config = (
        BitsAndBytesConfig(
            # load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )
        if quantize
        else None
    )

    # Try passing config as None, investigate tuple exception
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        dtype=torch.float16,
        token=HF_TOKEN,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    return model


def load_tokenizer(model_name: str | Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=TRAIN_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="qwen3-debiasing",
        dataloader_num_workers=2,
        fp16=True,
    )


def debias_text(text: str, model, tokenizer, max_length: int = 512):
    inputs = tokenizer(
        [f"{b}\n" for b in text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    predicted_texts = [
        d[len(b) :].strip() if d.startswith(b) else d.strip()
        for d, b in zip(decoded_outputs, text)
    ]

    return predicted_texts


def load_cache():
    if JUDGE_CACHE_FILE.exists():
        cache = {}
        with JUDGE_CACHE_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = entry["key"]
                cache[key] = entry
        return cache
    return {}


def save_to_cache(entry):
    with JUDGE_CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def make_cache_key(biased, prediction, neutral_ref):
    return f"{hash((biased, prediction, neutral_ref))}"
