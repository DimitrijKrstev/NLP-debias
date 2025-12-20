from logging import getLogger
from typing import Any, List

import torch
from torch.nn.functional import log_softmax
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from soft_label_distil.models import LogProbDTO
from soft_label_distil.utils import (
    build_teacher_distribution,
    get_teacher_logprobs,
)

logger = getLogger(__name__)


class DistillationTrainer(Trainer):
    processing_class: PreTrainedTokenizer
    teacher_model_name: str
    temperature: float
    alpha: float

    def __init__(
        self,
        model: Any,
        args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any,
        data_collator: Any,
        tokenizer: PreTrainedTokenizer,
        teacher_model_name: str = "qwen/qwen3-235b-a22b",
        temperature: float = 1.5,
        alpha: float = 0.5,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        self.processing_class = tokenizer
        self.teacher_model_name = teacher_model_name
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(
            **{
                input: value
                for input, value in inputs.items()
                if input in ["input_ids", "attention_mask", "labels"]
            }
        )
        student_logits = outputs.logits
        tokenized_inputs = inputs.get("tokenized_input", {})
        labels = inputs.get("labels", [])

        decoded_batched_biased_texts = self.processing_class.batch_decode(
            tokenized_inputs,
            skip_special_tokens=True,
        )

        standard_hard_loss = outputs.loss
        distillation_loss = 0.0
        try:
            batch_kl_losses = [
                kl_loss
                for i, biased_text in enumerate(decoded_batched_biased_texts)
                if (
                    kl_loss := self._compute_kl_divergence(
                        student_logits[i],
                        get_teacher_logprobs(biased_text, self.teacher_model_name),
                        labels[i],
                    )
                )
                or kl_loss == 0.0
            ]

            filtered_batch_kl_losses = [
                loss for loss in batch_kl_losses if loss is not None
            ]

            if len(filtered_batch_kl_losses) != len(decoded_batched_biased_texts):
                logger.error(
                    "Some KL losses could not be computed. Difference of %d",
                    len(decoded_batched_biased_texts) - len(filtered_batch_kl_losses),
                )

            if filtered_batch_kl_losses:
                kl_values = [kl.item() for kl in filtered_batch_kl_losses]
                logger.info(
                    f"Batch KL divergences: min={min(kl_values):.4f}, "
                    f"max={max(kl_values):.4f}, mean={sum(kl_values) / len(kl_values):.4f}, "
                    f"count={len(kl_values)}/{len(decoded_batched_biased_texts)}"
                )

                distillation_loss = torch.stack(filtered_batch_kl_losses).mean()
            else:
                logger.warning("[DEBUG] No KL losses were computed for this batch!")

        except Exception as e:
            logger.warning(f"Error computing distillation loss: {e}")

        if isinstance(distillation_loss, float) and distillation_loss == 0.0:
            logger.warning(
                "[DEBUG] Distillation loss is 0.0 for this batch - no teacher logprobs were successfully computed. "
                "Using only standard hard loss."
            )
            total_loss = standard_hard_loss
        else:
            total_loss = (
                self.alpha * distillation_loss + (1 - self.alpha) * standard_hard_loss
            )

            self.log(
                {
                    "distillation_loss": (
                        distillation_loss.item()
                        if torch.is_tensor(distillation_loss)
                        else distillation_loss
                    ),
                    "hard_loss": (
                        standard_hard_loss.item()
                        if torch.is_tensor(standard_hard_loss)
                        else standard_hard_loss
                    ),
                    "total_loss": (
                        total_loss.item() if torch.is_tensor(total_loss) else total_loss
                    ),
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logprobs: List[LogProbDTO],
        labels: torch.Tensor,
        coverage_threshold: float = 0.8,
    ) -> torch.Tensor | None:
        if not teacher_logprobs:
            logger.warning("Found missing teacher logprobs")
            return None

        prompt_length = 0
        for label in labels:
            if label == -100:
                prompt_length += 1
            else:
                break

        position_losses = []
        vocab_size = student_logits.shape[-1]
        low_coverage_count = 0

        for pos, teacher_token_logprob in enumerate(teacher_logprobs):
            if not teacher_token_logprob.top_logprobs:
                logger.warning(
                    f"No top logprobs for token {teacher_token_logprob.token}"
                )
                continue

            student_pos = prompt_length + pos
            if student_pos >= len(student_logits):
                logger.warning(
                    f"Student sequence too short: student_pos={student_pos} >= len={len(student_logits)}, "
                    f"only used {pos}/{len(teacher_logprobs)} teacher positions"
                )
                break

            teacher_probabilities, raw_coverage = build_teacher_distribution(
                teacher_token_logprob,
                vocab_size,
                self.processing_class,
                self.temperature,
                student_logits.device,
            )

            student_log_probs = log_softmax(
                student_logits[student_pos] / self.temperature, dim=-1
            )

            should_normalize_teacher_probs = raw_coverage >= coverage_threshold
            if should_normalize_teacher_probs:
                teacher_probabilities = (
                    teacher_probabilities / teacher_probabilities.sum()
                )

                # Prevent log(0)
                teacher_probabilities = teacher_probabilities.clamp_min(1e-8)

                kl = (
                    teacher_probabilities
                    * (torch.log(teacher_probabilities) - student_log_probs)
                ).sum()
            else:
                low_coverage_count += 1

                observed_tokens_mask = teacher_probabilities > 0
                cross_entropy = -(
                    teacher_probabilities[observed_tokens_mask]
                    * student_log_probs[observed_tokens_mask]
                ).sum()
                kl = cross_entropy

            position_losses.append(kl)

        if position_losses:
            if low_coverage_count > 0:
                logger.info(
                    f"Low coverage detected: {low_coverage_count}/{len(position_losses)} "
                    f"({low_coverage_count / len(position_losses):.1%}) positions used "
                    f"masked cross-entropy instead of KL divergence"
                )

            used_positions = len(position_losses)
            total_teacher_positions = len(teacher_logprobs)
            coverage = used_positions / total_teacher_positions

            if coverage < 0.5:
                logger.warning(
                    f"Low position coverage: only {used_positions}/{total_teacher_positions} "
                    f"({coverage:.1%}) teacher positions used. "
                    f"Student sequence may be too short."
                )

            mean_position_loss = torch.stack(position_losses).mean()
            temperature_scaled_position_losses = mean_position_loss * (
                self.temperature**2
            )

            return temperature_scaled_position_losses

        return None
