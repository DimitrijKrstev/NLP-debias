from dataclasses import dataclass


@dataclass
class Metrics:
    bleu: float
    meteor: float
    rouge1: float
    rouge2: float
    rougeL: float
    semantic_similarity: float
    perplexity: float

    @classmethod
    def from_scores(
        cls,
        bleu_score: dict | None,
        meteor_score: dict | None,
        rouge_scores: dict | None,
        perplexity_score: dict | None,
        semantic_sim: float,
    ) -> "Metrics":
        """Create Metrics from raw evaluation scores, handling scaling and missing values."""
        return cls(
            bleu=bleu_score["bleu"] * 100 if bleu_score else -1.0,
            meteor=meteor_score["meteor"] * 100 if meteor_score else -1.0,
            rouge1=rouge_scores["rouge1"] * 100 if rouge_scores else -1.0,
            rouge2=rouge_scores["rouge2"] * 100 if rouge_scores else -1.0,
            rougeL=rouge_scores["rougeL"] * 100 if rouge_scores else -1.0,
            semantic_similarity=semantic_sim * 100,
            perplexity=(
                perplexity_score["mean_perplexity"] if perplexity_score else -1.0
            ),
        )
