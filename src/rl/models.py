from pydantic import BaseModel, field_validator


class ModelResponseEvaluation(BaseModel):
    neutrality: int
    meaning_preservation: int
    fluency: int
    faithfulness_to_reference: int
    edit_minimality: int
    overall_reasoning: str

    @field_validator(
        "neutrality",
        "meaning_preservation",
        "fluency",
        "faithfulness_to_reference",
        "edit_minimality",
    )
    def validate_range(cls, value, field):
        if not -10 <= value <= 10:
            raise ValueError(f"{field.name} must be between -10 and 10, got {value}")
        return value

    def sum_all(self) -> float:
        return (
            self.neutrality
            + (self.meaning_preservation * 0.5)
            + (self.fluency * 0.2)
            + (self.faithfulness_to_reference * 0.5)
            + (self.edit_minimality * 0.2)
        )

    def normalized_score(self) -> float:
        max_possible = 1.0 * 10 + 0.5 * 10 + 0.2 * 10 + 0.5 * 10 + 0.2 * 10
        min_possible = 1.0 * -10 + 0.5 * -10 + 0.2 * -10 + 0.5 * -10 + 0.2 * -10
        raw_score = self.sum_all()
        return (raw_score - min_possible) / (max_possible - min_possible)
