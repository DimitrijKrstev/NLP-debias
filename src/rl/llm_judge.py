from logging import getLogger

import torch
from torch.nn import Module

logger = getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LLMJudgeRewardModel(Module):
    def __init__(self, judge_fn, openai_model, tokenizer, dataset):
        super().__init__()
        self.judge_fn = judge_fn
        self.openai_model = openai_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.current_index = 0

    def score_response(self, response_text, biased_text, reference_text):
        if not response_text:
            return 0.0

        return (
            self.judge_fn(
                biased_text,
                response_text,
                reference_text,
                self.openai_model,
            )
            or 0.0
        )

    def forward(self, input_ids):
        response_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        sample = self.dataset[self.current_index]
        biased_text = sample.get("biased_text", "")
        reference_text = sample.get("reference_text", "")

        reward = self.score_response(response_text, biased_text, reference_text)

        self.current_index += 1

        return torch.as_tensor(reward, device=device, dtype=torch.float16)
