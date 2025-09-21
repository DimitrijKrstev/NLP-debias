import os
from logging import getLogger
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

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
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        dtype=torch.float16,
        token=HF_TOKEN,
        device_map="auto",
        # device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    return model


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
