from dataclasses import dataclass
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ModelResponseEvaluation(BaseModel):
    neutrality: int
    meaning_perservation: int
    fluency: int
    faithfulness_to_reference: int
    edit_minimality: int
    overall_reasoning: str

    def sum_all(self) -> float:
        return (
            self.neutrality
            + (self.meaning_perservation * 0.5)
            + (self.fluency * 0.2)
            + (self.faithfulness_to_reference * 0.5)
            + (self.edit_minimality * 0.2)
        )
