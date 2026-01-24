from pydantic import BaseModel, field_validator


class ModelResponseEvaluation(BaseModel):
    neutrality: int
    meaning_preservation: int
    fluency: int
    edit_minimality: int
    overall_reasoning: str | None

    @field_validator(
        "neutrality",
        "meaning_preservation",
        "fluency",
        "edit_minimality",
    )
    def validate_range(cls, value, field):
        if not -3 <= value <= 3:
            raise ValueError(f"{field.name} must be between -3 and 3, got {value}")
        return value

    def get_normalized_full_score(self) -> float:
        return (
            self.neutrality
            + self.meaning_preservation
            + (self.fluency * 0.5)
            + (self.edit_minimality * 0.5)
        ) / 9.0
