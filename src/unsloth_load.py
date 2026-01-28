import os
from logging import getLogger
from typing import Any
from unsloth import FastLanguageModel # type: ignore

HF_TOKEN = os.getenv("HF_TOKEN")

logger = getLogger(__name__)

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