from torch import tensor
from torch.nn import Module

from judge.utils import split_into_sentences


class LLMJudgeRewardModel(Module):
    def __init__(self, judge_fn, openai_model, tokenizer, dataset=None):
        super().__init__()
        self.judge_fn = judge_fn
        self.openai_model = openai_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_index = 0

    def score_response(self, response_text, biased_text, reference_text):
        sentences = split_into_sentences(response_text)

        if not sentences:
            return 0.0

        sentence_scores = []
        for sentence in sentences:
            score = self.judge_fn(
                biased_text,
                sentence,
                reference_text,
                self.openai_model,
            )
            if score is not None:
                sentence_scores.append(score)

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

            reward = self.score_response(response_text, biased_text, reference_text)
            rewards.append(reward)

        self.batch_index += batch_size

        return tensor(rewards).unsqueeze(1)
