import os
from logging import getLogger
from pathlib import Path
from re import DOTALL, sub
from typing import Any

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from unsloth import FastLanguageModel  # type: ignore

HF_TOKEN = os.getenv("HF_TOKEN")

logger = getLogger(__name__)


def load_model(model_name: str | Path, quantize: bool) -> Any:
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quantize
        else None
    )

    logger.info(f"Loading model {model_name} with bnb config:\n{bnb_config}")
    # Try passing config as None, investigate tuple exception
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
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
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer


def load_peft_model(model_name: str, quantize: bool) -> Any:
    logger.info(f"Loading model: {model_name}")

    model = load_model(model_name, quantize)

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

    return model


def load_peft_classifcation_model(model_name: str, quantize: bool) -> Any:
    logger.info(f"Loading model for binary classification: {model_name}")

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if quantize
        else None
    )

    logger.info(f"Loading model {model_name} with bnb config:\n{bnb_config}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_labels = 2
    torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        num_labels=2,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False

    if quantize:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
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

    return model


def load_unsloth_model(model_name: str, quantize: bool) -> tuple[Any, Any]:
    logger.info(f"Loading model with Unsloth: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=quantize,
        token=HF_TOKEN,
    )

    logger.info("Applying LoRA configuration with Unsloth")
    model = FastLanguageModel.get_peft_model(
        model,
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
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    model.print_trainable_parameters()
    return model, tokenizer


def remove_thinking_tags(text: str) -> str:
    text = sub(r"<think>.*?</think>", "", text, flags=DOTALL)
    return text.strip()
