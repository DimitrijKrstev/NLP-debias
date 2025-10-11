import os
from logging import getLogger
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from constants import OUTPUT_DIR

HF_TOKEN = os.getenv("HF_TOKEN")


logger = getLogger(__name__)


def load_peft_model_and_tokenizer(model_name: str, quantize: bool) -> tuple[Any, Any]:
    logger.info(f"Loading model: {model_name}")

    model, tokenizer = get_model_and_tokenizer(model_name, quantize)

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


def get_model_and_tokenizer(model_name: str, quantize: bool) -> tuple[Any, Any]:
    model = load_model(model_name, quantize)
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


def load_model(model_name: str, quantize: bool) -> Any:
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
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
    # config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # config=config,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        max_steps=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=600,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["mlflow"],
        run_name="qwen3-debiasing",
    )


def debias_text(text: str, model, tokenizer, max_length: int = 512):
    inputs = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return generated_text.strip()
