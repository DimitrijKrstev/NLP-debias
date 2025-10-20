from evaluate import load as evaluate_load
from sentence_transformers import SentenceTransformer, util


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    bleu = evaluate_load("bleu")
    meteor = evaluate_load("meteor")
    rouge = evaluate_load("rouge")

    bleu_score = bleu.compute(predictions=predictions, references=references)
    meteor_score = meteor.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    pred_emb = similarity_model.encode(predictions, convert_to_tensor=True)
    ref_emb = similarity_model.encode(references, convert_to_tensor=True)
    semantic_sim = util.cos_sim(pred_emb, ref_emb).diagonal().mean().item()

    return {
        "bleu": bleu_score["bleu"] * 100,
        "meteor": meteor_score["meteor"] * 100,
        "rouge1": rouge_scores["rouge1"] * 100,
        "rouge2": rouge_scores["rouge2"] * 100,
        "rougeL": rouge_scores["rougeL"] * 100,
        "semantic_similarity": semantic_sim * 100,
    }
