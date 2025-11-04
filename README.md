# NLP Debiasing

This project explores automated text debiasing using both supervised fine-tuning (SFT) and reinforcement learning through Group Relative Policy Optimization (GRPO). The two training stages are conducted independently: the RL phase is applied directly to the base model rather than to the SFT-trained model. Using the Wikipedia Neutrality Corpus and LLM-as-a-Judge reward modeling, the system aims to neutralize biased text while preserving semantic content.

## Setup

Install dependencies using `uv`:

```bash
uv sync
```

Set up environment variables:

```bash
# Create .env file
echo "HF_TOKEN=your_huggingface_token" >> .env
echo "OPENAI_API_KEY=your_openai_key" >> .env
```

## Usage

```bash
uv run src/main.py download-dataset
```

Downloads the Wikipedia Neutrality Corpus (WNC) for model training and evaluation.

```bash
uv run src/main.py sft-train-model --model-name Qwen/Qwen3-4B --quantize
```

Fine-tunes a pretrained language model using the WNC dataset.

`--model-name` - The Hugging Face model name or checkpoint to use as the starting point for training.

`--quantize` - If enabled, loads the model in quantized mode (e.g., 4-bit or 8-bit), which reduces GPU memory usage and speeds up training.

`--mlflow-experiment` - The MLflow experiment name where metrics, logs, and model checkpoints will be stored.

```bash
uv run src/main.py train-rl-model --model-name Qwen/Qwen3-4B --quantize
```

Applies Reinforcement Learning from Human Feedback (RLHF) using Group Relative Policy Optimization (GRPO) to improve neutrality alignment.

```bash
uv run src/main.py eval-model --model-tokenizer-path output/ --model-name Qwen/Qwen3-4B
```

Evaluates the model’s performance on the evaluation set.

`--model-tokenizer-path` - Directory containing the fine-tuned model and tokenizer.

`--model-name` - The model’s architecture or name used for evaluation logs.
