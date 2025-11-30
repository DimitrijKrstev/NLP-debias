import os
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

HF_TOKEN = os.getenv("HF_TOKEN")


logger = getLogger(__name__)


def load_peft_model(model_name: str, quantize: bool = True) -> Any:
    logger.info(f"Loading model: {model_name}")
    model_path = Path(model_name)

    if model_path.exists() and (model_path / "adapter_config.json").exists():
        logger.info("Detected PEFT adapter checkpoint")

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

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            is_trainable=True,
            offload_buffers=True,
            offload_folder="offload",
        )

        model.gradient_checkpointing_disable()
        model.config.use_cache = False

        if quantize:
            model = prepare_model_for_kbit_training(model)

        model.print_trainable_parameters()
        return model

    # 🔹 CASE 2: initial SFT (fresh LoRA)
    model = load_model(model_name, quantize)

    if quantize:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    logger.info("Applying LoRA configuration (fresh)")
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer


def get_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir="./output",
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
        bf16=True,
    )
