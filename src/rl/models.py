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

    def get_normalized_full_score(self) -> float:
        return (
            self.neutrality
            + self.meaning_preservation
            + (self.fluency * 0.5)
            + self.faithfulness_to_reference
            + (self.edit_minimality * 0.5)
        ) / 40.0
