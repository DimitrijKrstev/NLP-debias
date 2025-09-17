from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class WNCData:
    input_text: str
    biased_text: str
    neutral_text: str

    def to_dict(self) -> dict:
        return {
            "input_text": self.input_text,
            "biased_text": self.biased_text,
            "neutral_text": self.neutral_text,
        }
