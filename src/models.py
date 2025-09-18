from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class WNCData:
    model_input_text: str
    biased: str
    neutral: str

    def to_dict(self) -> dict:
        return {
            "model_input_text": self.model_input_text,
            "biased": self.biased,
            "neutral": self.neutral,
        }
