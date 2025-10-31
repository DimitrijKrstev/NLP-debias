import torch
from torch.nn import Module
import typer
from logging import getLogger

from judge.utils import split_into_sentences

app = typer.Typer()

logger = getLogger(__name__)


class LLMJudgeRewardModel(Module):
    def __init__(self, judge_fn, openai_model, tokenizer, dataset=None):
        super().__init__()
        self.judge_fn = judge_fn
        self.openai_model = openai_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_batch_index(self):
        self.batch_index = 0

    def score_response(self, response_text, biased_text, reference_text):
        sentences = split_into_sentences(response_text)

        if not sentences:
            return 0.0

        sentence_scores = []
        for sentence in sentences:
            try:
                score = self.judge_fn(
                    biased_text=biased_text,
                    model_output=sentence,
                    reference_text=reference_text,
                    openai_model=self.openai_model,
                )
                if score is not None:
                    normalized = max(0.0, min(1.0, score / 10.0))
                    sentence_scores.append(normalized)
            except Exception as e:
                logger.warning(f"Error scoring sentence: {e}")
                continue

        if sentence_scores:
            return sum(sentence_scores) / len(sentence_scores)
        return 0.0

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        rewards = []

        for i in range(batch_size):
            response_text = self.tokenizer.decode(
                input_ids[i], skip_special_tokens=True
            )

            # Get context from dataset if available
            if self.dataset is not None and self.batch_index + i < len(self.dataset):
                sample = self.dataset[self.batch_index + i]
                biased_text = sample.get("biased_text", "")
                reference_text = sample.get("reference_text", "")
            else:
                biased_text = ""
                reference_text = ""
                logger.warning(f"No dataset context for sample {self.batch_index + i}")

            reward = self.score_response(response_text, biased_text, reference_text)
            rewards.append(reward)

        self.batch_index += batch_size

        return torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
